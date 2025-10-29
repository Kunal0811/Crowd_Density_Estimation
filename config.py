import numpy as np

# ----------------- CONFIGURATION -----------------

# Define the polygon for the Region of Interest (ROI)
# These are pixel coordinates (x, y) that you need to adjust for your camera view.
# Example: A trapezoid for a perspective view of a floor area.
ROI_CORNERS = np.array([
    [50, 450],   # Bottom-left
    [600, 450],  # Bottom-right
    [400, 200],  # Top-right
    [200, 200]   # Top-left
], np.int32)

# The real-world area of the ROI in square meters.
# You need to measure this area once in your physical setup.
ROI_AREA_M2 = 15.0 

# Confidence threshold for YOLOv8 model
# Detections with confidence below this value will be ignored.
MODEL_CONFIDENCE = 0.40