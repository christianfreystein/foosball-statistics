import cv2
from ultralytics import YOLO

# --- Configuration ---
# Load the YOLOv11 model
# model = YOLO("/home/freystec/repositories/yolo-distiller/runs/detect/train/weights/best.pt")
# model = YOLO("/home/freystec/foosball-statistics/weights/yolov11l_imgsz_640.pt")
model = YOLO("/home/freystec/foosball-statistics/weights/yolov8n_puppets_first_iteration.pt")
# Video paths
# input_video_path = "/home/freystec/foosball-statistics/foosball-videos/Leonhart_clip.mp4"
input_video_path = "/home/freystec/foosball-statistics/foosball-videos/Bundesliga_2024_Bamberg_Kicker_Crew_Bonn.mp4"
# input_video_path = "/home/freystec/foosball-statistics/foosball-videos/Westermann Bade VS Hoffmann Spredeman [1sBcrUBZxlY].mp4"
# input_video_path = "/home/freystec/foosball-statistics/foosball-videos/topview_leo_championship_match2.mp4"

output_video_path = "yolov8n_puppets_Bundesliga_Bamberg_Kicker_Crew_Bonn.mp4"

# Define the number of pixels to crop from each side
# You can change these values to adjust the cropping.
# For example, to cut 150 pixels from both left and right:
PIXELS_TO_CROP_LEFT = 300
PIXELS_TO_CROP_RIGHT = 300

# --- Video Processing ---
cap = cv2.VideoCapture(input_video_path)

# Check if video opened successfully
if not cap.isOpened():
    print(f"Error: Could not open video file {input_video_path}")
    exit()

# Get original video properties
fps = int(cap.get(cv2.CAP_PROP_FPS))
original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Calculate new dimensions after cropping
cropped_width = original_width - PIXELS_TO_CROP_LEFT - PIXELS_TO_CROP_RIGHT
cropped_height = original_height # Height remains unchanged

# Ensure new dimensions are valid
if cropped_width <= 0 or cropped_height <= 0:
    print(f"Error: Cropping dimensions result in invalid video size ({cropped_width}x{cropped_height}).")
    print(f"Original: {original_width}x{original_height}, Crop Left: {PIXELS_TO_CROP_LEFT}, Crop Right: {PIXELS_TO_CROP_RIGHT}")
    exit()

# Set up VideoWriter for the output video
fourcc = cv2.VideoWriter_fourcc(*"mp4v") # You can also try 'XVID' or 'MJPG' if 'mp4v' causes issues
out = cv2.VideoWriter(output_video_path, fourcc, fps, (cropped_width, cropped_height))

print(f"Processing video from {input_video_path}...")
print(f"Original dimensions: {original_width}x{original_height}")
print(f"Cropping {PIXELS_TO_CROP_LEFT} pixels from left and {PIXELS_TO_CROP_RIGHT} pixels from right.")
print(f"Output video dimensions: {cropped_width}x{cropped_height}")

frame_count = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break # End of video

    # --- Apply Cropping ---
    # Crop the frame from the sides
    # [height_start:height_end, width_start:width_end]
    
    cropped_frame = frame[0:original_height, PIXELS_TO_CROP_LEFT:original_width - PIXELS_TO_CROP_RIGHT]

    # Make YOLO predictions on the cropped frame
    # Note: If your model was trained on the full frame, cropping BEFORE inference
    # might reduce accuracy for objects near the edges.
    # If trained on cropped frames, this is correct.
    results = model(cropped_frame, verbose=False)

    # Draw bounding boxes on the cropped frame
    for result in results:
        for box in result.boxes:
            # Coordinates are relative to the 'cropped_frame'
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = box.conf[0].item()
            label = result.names[int(box.cls[0].item())]

            # Draw rectangle and label on the cropped frame
            cv2.rectangle(cropped_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(cropped_frame, f"{label} {conf:.2f}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Write the cropped and annotated frame to the output video
    out.write(cropped_frame)
    frame_count += 1

print(f"Processed {frame_count} frames.")

# Release resources
cap.release()
out.release()
# cv2.destroyAllWindows() # Good practice, though no windows are explicitly opened here
print("Processing complete. Output saved to", output_video_path)