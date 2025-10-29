import cv2
import os
import numpy as np
import json

# === Configuration ===
OUT = os.path.join(os.path.dirname(__file__), '..', 'calibration', 'calibration.json')
MOBILE_CAM_URL = "https://10.132.69.151:8080/video"  # Your mobile IP webcam URL
ARUCO_MARKER_SIZE_M = 0.2  # Marker size in meters (20 cm)

clicked = []


def mouse_cb(event, x, y, flags, param):
    """Capture 4 corner points with mouse clicks."""
    global clicked
    if event == cv2.EVENT_LBUTTONDOWN:
        clicked.append((x, y))
        print(f"Point clicked: {(x, y)}")


def detect_aruco_scale(frame):
    """
    Detect an ArUco marker and compute meters-per-pixel scale.
    Returns scale in meters/pixel or None if no marker detected.
    """
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    detector = cv2.aruco.ArucoDetector(aruco_dict)

    corners, ids, _ = detector.detectMarkers(frame)
    if ids is None or len(ids) == 0:
        return None  # No marker detected

    # Take the first detected marker
    c = corners[0][0]  # corners[0] is 4x2
    pixel_width = np.linalg.norm(c[0] - c[1])
    pixel_height = np.linalg.norm(c[1] - c[2])
    avg_pixel_size = (pixel_width + pixel_height) / 2

    scale = ARUCO_MARKER_SIZE_M / avg_pixel_size
    return scale


def main():
    global clicked
    cam = cv2.VideoCapture(MOBILE_CAM_URL)
    if not cam.isOpened():
        raise RuntimeError(f"Cannot open mobile camera: {MOBILE_CAM_URL}")

    window_name = 'calibrate'
    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, mouse_cb)

    print("\n=== Crowd Calibration ===")
    print("Click 4 corners of floor ROI.")
    print("[r] Reset points   [s] Save calibration   [Esc] Cancel\n")

    while True:
        ret, frame = cam.read()
        if not ret:
            continue

        display = frame.copy()
        for pt in clicked:
            cv2.circle(display, pt, 6, (0, 255, 0), -1)
        if len(clicked) == 4:
            cv2.polylines(display, [np.array(clicked)], isClosed=True, color=(255, 0, 0), thickness=2)

        cv2.imshow(window_name, display)
        key = cv2.waitKey(10) & 0xFF

        if key == ord('r'):
            clicked = []
            print("Points reset.")

        elif key == ord('s'):
            if len(clicked) != 4:
                print("⚠️ Need exactly 4 points!")
                continue

            # --- Try ArUco detection first ---
            scale = detect_aruco_scale(frame)
            if scale is None:
                print("⚠️ No ArUco marker detected!")
                try:
                    manual_scale = float(input("Enter manual scale (meters per pixel, e.g., 0.001): "))
                except Exception:
                    print("Invalid input. Using default 0.001 m/pixel.")
                    manual_scale = 0.001
                scale = manual_scale

            # --- Compute areas ---
            pts = np.array(clicked, dtype=np.int32)
            pixel_area = cv2.contourArea(pts)
            real_area = pixel_area * (scale ** 2)

            payload = {
                "points": clicked,
                "pixel_area": float(pixel_area),
                "real_area_m2": float(real_area),
                "scale_m2_per_pixel": float(scale ** 2)
            }

            os.makedirs(os.path.dirname(OUT), exist_ok=True)
            with open(OUT, 'w') as f:
                json.dump(payload, f, indent=2)

            print(f"✅ Calibration saved to {OUT}")
            break

        elif key == 27:  # Esc
            print("Calibration cancelled.")
            break

    cam.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
