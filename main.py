import os
import cv2
import time
import threading
import json
import pandas as pd
from datetime import datetime
import sys
import numpy as np
from queue import Queue, Empty
from pushbullet import Pushbullet
from detection.yolo_detector import YoloPersonDetector
from detection.density_model import DensityModel # This is unused now, but left for your reference
from webapp.app import run_app

# --- New Configuration ---
LOG_CSV = os.path.join('logs', 'density_logs.csv')
MOBILE_CAM_URL = "https://192.168.0.103:8080/video"  # !!! REMEMBER TO CHANGE THIS !!!
ALERT_COOLDOWN_SEC = 30  # Cooldown between push notifications
LEARNING_PERIOD_SEC = 90 # How long to "learn" the normal crowd level
ALERT_MULTIPLIER = 2.5   # Alert if count is 2.5x the average
# -------------------------

# --- Pushbullet API Key (No changes) ---
API_KEY = os.environ.get("PUSHBULLET_API_KEY")
if not API_KEY:
    print("ERROR: environment variable PUSHBULLET_API_KEY is not set.\n"
          "Set it and re-run the script. Example (PowerShell):\n"
          "$env:PUSHBULLET_API_KEY = 'o.xxxxx' ; python main.py")
    sys.exit(1)
pb = Pushbullet(API_KEY)
last_alert_time = 0
# -------------------------


# --- Shared state for Flask stream ---
shared_state = {
    'last_frame': None,
    'last_count': 0,
    'alert_threshold': -1 # -1 means 'calibrating'
}


def overlay_info(frame, boxes, count, alert_threshold):
    """Draws boxes and info on the frame."""
    global last_alert_time
    display_frame = frame.copy()

    # Draw boxes
    for (x1, y1, x2, y2) in boxes:
        cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
    
    # Draw Count
    cv2.putText(display_frame, f'Count: {count}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255,255,255), 2)

    # Status Text
    if alert_threshold == -1:
        # We are in the learning phase
        cv2.putText(display_frame, 'CALIBRATING...', (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
    else:
        # We are in normal operation
        cv2.putText(display_frame, f'Alert Threshold: {alert_threshold}', (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,255,255), 2)
        
        # Check for alert condition
        if count > alert_threshold:
            cv2.putText(display_frame, 'ALERT! Overcrowded', (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,255), 3)
            
            # Send Pushbullet notification if cooldown has passed
            current_time = time.time()
            if (current_time - last_alert_time) > ALERT_COOLDOWN_SEC:
                try:
                    pb.push_note("Overcrowding Alert", f"Crowd count is {count}, which is over the threshold of {alert_threshold}")
                    last_alert_time = current_time
                    print("!!! ALERT SENT VIA PUSHBULLET !!!")
                except Exception as e:
                    print(f"Failed to send Pushbullet alert: {e}")

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


def process_loop():
    """
    Main processing loop with automatic threshold calibration.
    """
    detector = YoloPersonDetector()

    os.makedirs('logs', exist_ok=True)
    if not os.path.exists(LOG_CSV):
        pd.DataFrame(columns=['timestamp','count']).to_csv(LOG_CSV, index=False)

    frame_queue: Queue = Queue(maxsize=4)
    stop_event = threading.Event()

    # --- Frame Capture Thread (No changes) ---
    def capture_thread_fn(url, q: Queue, stop_evt: threading.Event):
        """Continuously capture frames and put into queue (drops frames when full)."""
        while not stop_evt.is_set():
            cap = cv2.VideoCapture(url)
            if not cap.isOpened():
                print(f"Error: Cannot open camera at {url}. Retrying in 1s...")
                time.sleep(1.0)
                continue

            print("Camera connection successful.")
            while not stop_evt.is_set():
                ret, frame = cap.read()
                if not ret:
                    print("Frame read error, stream may have disconnected. Reconnecting...")
                    time.sleep(0.05)
                    break # Break inner loop to trigger reconnect

                try:
                    if q.full():
                        _ = q.get_nowait()
                    q.put_nowait(frame)
                except Empty:
                    pass
                except Exception:
                    pass
            try:
                cap.release()
            except Exception:
                pass
            time.sleep(0.5)

    cap_thread = threading.Thread(target=capture_thread_fn, args=(MOBILE_CAM_URL, frame_queue, stop_event), daemon=True)
    cap_thread.start()

    # --- New Automatic Threshold Logic ---
    start_time = time.time()
    learning_counts = []
    is_learning = True
    alert_threshold_count = -1 # -1 signifies 'learning'
    # -----------------------------------

    last_log_time = time.time()

    print(f"--- Starting {LEARNING_PERIOD_SEC} second calibration phase... ---")

    while True:
        try:
            frame = frame_queue.get(timeout=2.0)
        except Empty:
            print("Queue empty, no frame received.")
            time.sleep(0.1)
            continue

        # Resize for faster processing
        proc_w = 640
        h, w = frame.shape[:2]
        if w > proc_w:
            scale = proc_w / float(w)
            proc_h = int(h * scale)
            proc_frame = cv2.resize(frame, (proc_w, proc_h))
        else:
            proc_frame = frame.copy()

        try:
            # We removed the ROI, so detection runs on the whole frame
            boxes = detector.detect(proc_frame)
        except Exception as e:
            print(f"Detector error: {e}")
            boxes = []

        # Scale boxes back to original frame size
        if proc_frame.shape[1] != w or proc_frame.shape[0] != h:
            sx = w / float(proc_frame.shape[1])
            sy = h / float(proc_frame.shape[0])
            scaled_boxes = []
            for (x1, y1, x2, y2) in boxes:
                scaled_boxes.append((int(x1 * sx), int(y1 * sy), int(x2 * sx), int(y2 * sy)))
            boxes = scaled_boxes

        count = len(boxes)

        # --- Handle Learning Phase ---
        if is_learning:
            learning_counts.append(count)
            elapsed = time.time() - start_time
            print(f"Calibrating... {int(elapsed)}/{LEARNING_PERIOD_SEC}s. Current count: {count}")
            
            if elapsed >= LEARNING_PERIOD_SEC:
                if not learning_counts:
                    print("WARNING: Calibration finished but no counts recorded. Setting threshold to 10.")
                    alert_threshold_count = 10
                else:
                    avg_count = np.mean(learning_counts)
                    std_dev = np.std(learning_counts)
                    # Set threshold to average + (multiplier * std_dev), or just avg * multiplier
                    # Using a simple multiplier of the average is more robust to outliers
                    calculated_threshold = avg_count * ALERT_MULTIPLIER
                    
                    # Ensure threshold is at least a reasonable minimum (e.g., 5)
                    alert_threshold_count = max(5, int(calculated_threshold))

                print("--- CALIBRATION COMPLETE ---")
                print(f"Avg count: {np.mean(learning_counts):.2f}")
                print(f"New Alert Threshold (Count): {alert_threshold_count}")
                print("------------------------------")
                is_learning = False
        
        # Annotate the original frame for streaming
        annotated = overlay_info(frame, boxes, count, alert_threshold_count)

        # Update shared state for web app
        shared_state['last_frame'] = annotated.copy()
        shared_state['last_count'] = count
        shared_state['alert_threshold'] = alert_threshold_count # Send threshold to web app

        # Log only every 5 seconds
        if time.time() - last_log_time >= 5:
            ts = datetime.utcnow().isoformat()
            # Log only count, no density
            pd.DataFrame([{'timestamp': ts, 'count': count}]).to_csv(
                LOG_CSV, mode='a', header=False, index=False)
            last_log_time = time.time()

        time.sleep(0.02)

if __name__ == "__main__":
    # We no longer load calibration!
    # calib = load_calibration() 

    # Inject frame generator into Flask app
    from webapp import app as flask_app
    import webapp.app as webmodule
    webmodule.frame_generator = frame_stream

    # Start Flask in a separate thread
    flask_thread = threading.Thread(target=run_app, kwargs={'port':5000}, daemon=True)
    flask_thread.start()
    print("Flask running at http://127.0.0.1:5000")

    # Start processing
    process_loop() # No 'calib' argument needed