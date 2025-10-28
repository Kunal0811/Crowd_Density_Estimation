import os
import cv2
import sys
import numpy as np
from detection.yolo_detector import YoloPersonDetector

# This helper function is copied from main.py
def overlay_info_static(frame, boxes):
    """Draws boxes and info on a static frame."""
    display_frame = frame.copy()
    count = len(boxes)
    
    h, w = display_frame.shape[:2]

    # --- UPDATED TEXT ---
    
    # 1. Set font properties
    text = f'Total Count: {count}'
    font_face = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 2.0  # Increased font scale
    thickness = 3
    
    # 2. Get text size to create a background
    (text_w, text_h), baseline = cv2.getTextSize(text, font_face, font_scale, thickness)
    
    # 3. Draw a black background rectangle at the bottom
    # Make the rectangle 20px taller than the text for padding
    rect_h = text_h + 20
    cv2.rectangle(display_frame, (0, h - rect_h), (w, h), (0, 0, 0), -1) # Solid black
    
    # 4. Draw the white text on top of the background
    # Centered horizontally, and positioned 10px from the bottom
    text_x = (w - text_w) // 2
    text_y = h - 15 # 15 pixels up from the bottom edge
    
    cv2.putText(display_frame, text, (text_x, text_y), 
                font_face, font_scale, (255, 255, 255), thickness)

    # --- END UPDATED TEXT ---

    # Draw boxes
    for (x1, y1, x2, y2) in boxes:
        cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    return display_frame

def process_image(detector, input_path, output_path):
    """Detects people in a single image and saves the result."""
    print(f"Processing image: {input_path}")
    frame = cv2.imread(input_path)
    if frame is None:
        print(f"Error: Could not read image from {input_path}")
        return

    # Run detection
    boxes = detector.detect(frame)
    
    # Annotate frame
    annotated_frame = overlay_info_static(frame, boxes)
    
    # Save result
    cv2.imwrite(output_path, annotated_frame)
    print(f"Result saved to: {output_path}")

def process_video(detector, input_path, output_path):
    """Detects people in a video file and saves the result."""
    print(f"Processing video: {input_path}")
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file {input_path}")
        return

    # Get video properties for output
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    # Define the codec and create VideoWriter object
    # Use 'mp4v' for .mp4 files
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        frame_count += 1
        print(f"Processing frame {frame_count}...")

        # Run detection
        boxes = detector.detect(frame)
        
        # Annotate frame
        annotated_frame = overlay_info_static(frame, boxes)
        
        # Write the processed frame
        out.write(annotated_frame)

    cap.release()
    out.release()
    print(f"Video result saved to: {output_path}")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python process_file.py <input_file_path> <output_file_path>")
        sys.exit(1)
        
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    
    # Initialize the detector
    try:
        detector = YoloPersonDetector()
    except Exception as e:
        print(f"Error initializing YOLO detector: {e}")
        print("Please ensure your 'models/yolo_weights.pt' file exists.")
        sys.exit(1)

    # Check file type
    ext = os.path.splitext(input_file)[1].lower()
    if ext in ['.jpg', '.jpeg', '.png']:
        process_image(detector, input_file, output_file)
    elif ext in ['.mp4', '.avi', '.mov', '.mkv']:
        process_video(detector, input_file, output_file)
    else:
        print(f"Error: Unsupported file type: {ext}")
        sys.exit(1)
