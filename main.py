import os
import cv2
import time
import threading
import json
from flask import jsonify
import pandas as pd
from datetime import datetime
import sys
import numpy as np
from queue import Queue, Empty
from pushbullet import Pushbullet
from detection.yolo_detector import YoloPersonDetector
from detection.density_model import DensityModel
from webapp.app import run_app

# Configuration
CALIB_PATH = os.path.join(os.path.dirname(__file__), 'calibration', 'calibration.json')
LOG_CSV = os.path.join('logs', 'density_logs.csv')
THRESHOLD_PPL_PER_M2 = 0.6   # Example threshold
ALERT_COOLDOWN_SEC = 30
MOBILE_CAM_URL = "http://10.16.213.107:8080/video"

# Correct way:
API_KEY = "o.5zicoAj1z2qs1tr3CdbxsN0xCBQe1mLx"
if not API_KEY:
    print("ERROR: environment variable PUSHBULLET_API_KEY is not set.\n"
          "Set it and re-run the script. Example (PowerShell):\n"
          "$env:PUSHBULLET_API_KEY = 'o.xxxxx' ; python main.py")
    sys.exit(1)


# Shared state for Flask stream
shared_state = {
    'last_frame': None,
    'last_count': 0,
    'last_density': 0.0
}


def load_calibration():
    if not os.path.exists(CALIB_PATH):
        raise FileNotFoundError('Calibration file not found. Run calibration/calibrate.py first.')
    with open(CALIB_PATH, "r") as f:
        data = json.load(f)
    return data


def overlay_info(frame, boxes, count, density, threshold, calib):
    # Resize frame for display only (keeps processing original size)
    display_frame = frame.copy()
    if 'points' in calib:
        pts = [(int(x), int(y)) for x, y in calib['points']]
        cv2.polylines(display_frame, [np.array(pts, np.int32)], True, (255, 0, 0), 2)
    for (x1, y1, x2, y2) in boxes:
        cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.putText(display_frame, f'Count: {count}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255,255,255), 2)
    cv2.putText(display_frame, f'Density: {density:.2f} p/m2', (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,255,255), 2)
    if density > threshold:
        pb = Pushbullet(API_KEY)
        pb.push_note("Alert", "Overcrowded!")

        cv2.putText(display_frame, 'ALERT! Overcrowded', (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,255), 3)
    return display_frame  # Return the annotated frame


def frame_stream():
    """Generator for MJPEG streaming"""
    while True:
        frame = shared_state['last_frame']
        if frame is None:
            time.sleep(0.1)
            continue
        # Resize for streaming (reduce lag and bandwidth). Keep original frame untouched.
        stream_frame = cv2.resize(frame, (640, 360))  # adjust smaller if needed

        # Lower JPEG quality to reduce CPU and bandwidth
        ret, jpeg = cv2.imencode('.jpg', stream_frame, [int(cv2.IMWRITE_JPEG_QUALITY), 50])
        if ret:
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n')

        # Stream at ~5 FPS to reduce CPU on slower machines
        time.sleep(0.2)



def process_loop(calib):
    """
    Start a capture thread that fills a small queue and a processing loop that consumes frames.
    This decouples network/capture jitter from the heavier detection work and streaming.
    """
    detector = YoloPersonDetector()
    dmodel = DensityModel()  # optional

    os.makedirs('logs', exist_ok=True)
    if not os.path.exists(LOG_CSV):
        pd.DataFrame(columns=['timestamp','count','density']).to_csv(LOG_CSV, index=False)

    frame_queue: Queue = Queue(maxsize=4)
    stop_event = threading.Event()

    def capture_thread_fn(url, q: Queue, stop_evt: threading.Event):
        """Continuously capture frames and put into queue (drops frames when full)."""
        while not stop_evt.is_set():
            cap = cv2.VideoCapture(url)
            if not cap.isOpened():
                # failed to open, wait and retry
                time.sleep(1.0)
                continue

            while not stop_evt.is_set():
                ret, frame = cap.read()
                if not ret:
                    time.sleep(0.05)
                    # try reopening the stream
                    break

                try:
                    # If queue is full, drop the oldest frame to keep latency low
                    if q.full():
                        try:
                            _ = q.get_nowait()
                        except Empty:
                            pass
                    q.put_nowait(frame)
                except Exception:
                    # If any issue with queue, just continue
                    pass

            try:
                cap.release()
            except Exception:
                pass
            # brief pause before reconnect
            time.sleep(0.5)

    cap_thread = threading.Thread(target=capture_thread_fn, args=(MOBILE_CAM_URL, frame_queue, stop_event), daemon=True)
    cap_thread.start()

    last_log_time = time.time()  # for 5-second logging interval

    while True:
        try:
            frame = frame_queue.get(timeout=2.0)
        except Empty:
            # no frame available; continue and let stream show last frame
            time.sleep(0.1)
            continue

        # To speed up detection, run detector on a resized copy and then scale boxes back
        # Choose a reasonable processing width (smaller => faster, but lower accuracy)
        proc_w = 640
        h, w = frame.shape[:2]
        if w > proc_w:
            scale = proc_w / float(w)
            proc_h = int(h * scale)
            proc_frame = cv2.resize(frame, (proc_w, proc_h))
        else:
            proc_frame = frame.copy()

        # If calibration ROI exists, scale it to the processing frame size
        scaled_roi = None
        if 'points' in calib and calib['points']:
            # calib points are in original frame coordinates; scale them to proc_frame
            orig_h, orig_w = frame.shape[:2]
            proc_h, proc_w = proc_frame.shape[:2]
            sx = proc_w / float(orig_w)
            sy = proc_h / float(orig_h)
            scaled_roi = [(int(x * sx), int(y * sy)) for x, y in calib['points']]

        try:
            boxes = detector.detect(proc_frame, roi_polygon=scaled_roi)
        except Exception as e:
            # Log the error and continue with an empty detection result so the app stays responsive
            print(f"Detector error: {e}")
            boxes = []

        # If detection ran on a resized image, scale boxes back to original coords
        if proc_frame.shape[1] != w or proc_frame.shape[0] != h:
            sx = w / float(proc_frame.shape[1])
            sy = h / float(proc_frame.shape[0])
            scaled_boxes = []
            for (x1, y1, x2, y2) in boxes:
                scaled_boxes.append((int(x1 * sx), int(y1 * sy), int(x2 * sx), int(y2 * sy)))
            boxes = scaled_boxes

        count = len(boxes)
        real_area = float(calib.get('real_area_m2', 1.0))
        density = count / real_area

        # Annotate the original frame for streaming
        annotated = overlay_info(frame, boxes, count, density, THRESHOLD_PPL_PER_M2, calib)

        shared_state['last_frame'] = annotated.copy()
        shared_state['last_count'] = count
        shared_state['last_density'] = density

        # Emit a small console log for debugging so you can see counts in the terminal
        print(f"Updated: count={count}, density={density:.2f}")

        # Log only every 5 seconds
        if time.time() - last_log_time >= 5:
            ts = datetime.utcnow().isoformat()
            pd.DataFrame([{'timestamp': ts, 'count': count, 'density': density}]).to_csv(
                LOG_CSV, mode='a', header=False, index=False)
            last_log_time = time.time()

        # small sleep to yield - detection itself will dominate CPU; keep minimal
        time.sleep(0.02)

if __name__ == "__main__":
    calib = load_calibration()

    # Inject frame generator into Flask app
    from webapp import app as flask_app
    import webapp.app as webmodule
    webmodule.frame_generator = frame_stream

    # Start Flask in a separate thread
    flask_thread = threading.Thread(target=run_app, kwargs={'port':5000}, daemon=True)
    flask_thread.start()
    print("Flask running at http://127.0.0.1:5000")

    # Start processing
    process_loop(calib)