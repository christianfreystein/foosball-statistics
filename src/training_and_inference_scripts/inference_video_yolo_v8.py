import json
import os
import time
import contextlib
import sys
from collections import defaultdict

import cv2
from ultralytics import YOLO
from moviepy import VideoFileClip
import numpy as np # Import numpy for array operations

# Define paths
model_path = "/home/freystec/repositories/yolo-distiller/runs/detect/train/weights/yolov11n_with_KD_cropped.pt"
video_path = "/home/freystec/foosball-statistics/foosball-videos/Leonhart_clip.mp4"
annotated_video_path = "output_video.avi"
final_output_video_path = "test.mp4"
json_path = final_output_video_path.replace(".mp4", ".json")
temp_dir = "old/temp_predictions"

# Create the temporary directory for saving intermediate predictions
if not os.path.exists(temp_dir):
    os.makedirs(temp_dir)

# Load the YOLOv8 model
model = YOLO(model_path)

# Open the video file
cap = cv2.VideoCapture(video_path)

# Get properties of the original video
fps = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

# --- Define Crop Parameters ---
# Adjust these values based on your video to focus on the foosball table
crop_left = 150  # pixels to cut from the left
crop_right = 150 # pixels to cut from the right
crop_top = 0     # pixels to cut from the top
crop_bottom = 0  # pixels to cut from the bottom

# Calculate the dimensions of the cropped region for prediction
cropped_width = width - crop_left - crop_right
cropped_height = height - crop_top - crop_bottom

# Initialize video writer
fourcc = cv2.VideoWriter_fourcc(*'XVID')
# The output video writer should still use the original dimensions
out = cv2.VideoWriter(annotated_video_path, fourcc, fps, (width, height))

# Initialize frame counter
frame_count = 0

# List to store all predictions
predictions = []

# Start time for processing
start_time = time.time()

# Loop through the video frames
while cap.isOpened():
    # Read a frame from the video
    success, frame = cap.read()

    if success:
        # --- Crop the frame for prediction ---
        cropped_frame = frame[crop_top : height - crop_bottom, crop_left : width - crop_right]

        # Suppress YOLOv8 verbose output during inference
        with contextlib.redirect_stdout(open(os.devnull, 'w')):
            results = model(cropped_frame, verbose=False) # Perform inference on the cropped frame

        # Extract additional information
        inference_speed = results[0].speed

        # Get the annotated frame from the results of the cropped frame
        # We will re-draw the bounding boxes on the original frame later
        # to maintain the original video size for the output.
        
        # Initialize the frame's prediction data
        frame_data = {
            "frame_count": frame_count,
            "inference_speed": inference_speed,
            "boxes": []
        }

        # Create a copy of the original frame to draw annotations on
        annotated_frame = frame.copy()

        # Extract predictions and adjust coordinates
        for result in results[0].boxes:
            # Bounding box in original cropped coordinates (xywhn)
            bbox_normalized_cropped = result.xywhn[0].cpu().numpy()
            conf = float(result.conf[0].cpu().numpy())  # Confidence
            cls = int(result.cls[0].cpu().numpy())  # Class

            # Convert normalized (xywh) in cropped frame to absolute (xyxy) in cropped frame
            x_center_cropped, y_center_cropped, w_cropped, h_cropped = bbox_normalized_cropped
            
            # Calculate absolute pixel coordinates in the cropped frame
            x1_cropped = int((x_center_cropped - w_cropped / 2) * cropped_width)
            y1_cropped = int((y_center_cropped - h_cropped / 2) * cropped_height)
            x2_cropped = int((x_center_cropped + w_cropped / 2) * cropped_width)
            y2_cropped = int((y_center_cropped + h_cropped / 2) * cropped_height)

            # Adjust bounding box coordinates to the original frame's dimensions
            x1_original = x1_cropped + crop_left
            y1_original = y1_cropped + crop_top
            x2_original = x2_cropped + crop_left
            y2_original = y2_cropped + crop_top
            
            # Convert back to normalized xywh for saving
            x_center_original_norm = (x1_original + x2_original) / (2 * width)
            y_center_original_norm = (y1_original + y2_original) / (2 * height)
            w_original_norm = (x2_original - x1_original) / width
            h_original_norm = (y2_original - y1_original) / height

            bbox_original_normalized = [x_center_original_norm, y_center_original_norm, w_original_norm, h_original_norm]


            # Append the box data to the frame's prediction data
            frame_data["boxes"].append({
                "bbox": bbox_original_normalized, # Save adjusted bbox
                "conf": conf,
                "cls": cls
            })

            # Draw bounding box and label on the original frame for visualization
            # You can customize colors and thickness
            color = (0, 255, 0) # Green for bounding boxes
            thickness = 2
            cv2.rectangle(annotated_frame, (x1_original, y1_original), (x2_original, y2_original), color, thickness)
            
            # Optionally, add class name and confidence
            label = f"Class {cls}: {conf:.2f}"
            cv2.putText(annotated_frame, label, (x1_original, y1_original - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)


        # Write the annotated frame (with original dimensions) to the output video
        out.write(annotated_frame)

        # Append the frame's prediction data to the overall predictions
        predictions.append(frame_data)

        # Increment the frame counter
        frame_count += 1

        # Provide progress update
        if frame_count == 5 or frame_count % 100 == 0:
            elapsed_time = time.time() - start_time
            frames_processed = frame_count
            total_time_estimate = (elapsed_time / frames_processed) * total_frames
            remaining_time = (total_time_estimate - elapsed_time) / 60  # Convert to minutes
            print(f"Processed {frames_processed}/{total_frames} frames. "
                  f"Estimated time remaining: {remaining_time:.2f} minutes.")

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        # Break the loop if the end of the video is reached
        break

# Release the video capture and writer objects
cap.release()
out.release()
cv2.destroyAllWindows()

# Save predictions to a JSON file
with open(json_path, "w") as f:
    json.dump(predictions, f, indent=4)

# Merge video with original audio
processed_video = VideoFileClip(annotated_video_path)
original_video = VideoFileClip(video_path)

# Set the audio of the processed video to be the audio of the original video
processed_video = processed_video.set_audio(original_video.audio)

# Save the result
processed_video.write_videofile(final_output_video_path, codec='libx264', audio_codec='aac')

print(f"Predictions saved to {json_path}")
print(f"Video processing completed in {(time.time() - start_time) / 60:.2f} minutes.")
