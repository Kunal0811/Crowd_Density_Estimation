import os
import cv2
import numpy as np
from ultralytics import YOLO


MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', 'models', 'yolo_weights.pt')


class YoloPersonDetector:
    def __init__(self, model_path: str = MODEL_PATH, conf_thresh: float = 0.35, device: str = 'cpu'):
        self.conf_thresh = conf_thresh
        # if model file missing, user should download and put into models/
        if not os.path.exists(model_path):
            raise FileNotFoundError(f'YOLO weights not found at {model_path}. Place model there or change path.')

        self.model = YOLO(model_path)
        # set device if supported by this ultralytics build
        try:
            self.model.overrides['device'] = device
        except Exception:
            pass

    def detect(self, frame: np.ndarray, roi_polygon=None):
        """Runs detection on the frame. Returns list of boxes [(x1,y1,x2,y2), ...] for class 'person'.
        If roi_polygon provided (list of pts), detections outside polygon are discarded.
        """
        # ultralytics expects RGB
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.model.predict(source=img, conf=self.conf_thresh, classes=[0], verbose=False)

        boxes = []
        if len(results) > 0:
            res = results[0]
            if hasattr(res, 'boxes') and len(res.boxes) > 0:
                # res.boxes.data may be a tensor; try to convert to python list robustly
                rows = []
                try:
                    rows = res.boxes.data.cpu().numpy().tolist()
                except Exception:
                    try:
                        rows = res.boxes.data.tolist()
                    except Exception:
                        # fall back to iterating box objects
                        for box in res.boxes:
                            try:
                                coords = box.xyxy[0].tolist()
                                rows.append(coords)
                            except Exception:
                                pass

                for b in rows:
                    # expected format: [x1, y1, x2, y2, conf, cls] or similar
                    if len(b) < 4:
                        continue
                    x1, y1, x2, y2 = map(int, b[:4])
                    rect = (x1, y1, x2, y2)

                    if roi_polygon is not None:
                        cx = int((x1 + x2) / 2)
                        cy = int((y1 + y2) / 2)
                        if not point_in_poly((cx, cy), roi_polygon):
                            continue

                    boxes.append(rect)

        return boxes


def point_in_poly(pt, poly):
    """Simple point in polygon test using cv2.pointPolygonTest"""
    poly_np = np.array(poly, dtype=np.int32)
    return cv2.pointPolygonTest(poly_np, (int(pt[0]), int(pt[1])), False) >= 0


if __name__ == '__main__':
    # quick test
    det = YoloPersonDetector()
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        boxes = det.detect(frame)
        for (x1, y1, x2, y2) in boxes:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.imshow('YOLO test', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()