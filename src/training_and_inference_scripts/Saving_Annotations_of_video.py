import json
import os
import time
import contextlib
import sys
from collections import defaultdict

import cv2
from ultralytics import YOLO
from moviepy import VideoFileClip
import numpy as np

# Import supervision and tqdm libraries
import supervision as sv
from tqdm import tqdm # Import tqdm

# Define paths
model_path = "/home/freystec/repositories/yolo-distiller/runs/detect/train/weights/yolov11n_with_KD_cropped.pt"
# model_path = "/home/freystec/foosball-statistics/weights/yolov11l_imgsz_640.pt"
# video_path = "/home/freystec/foosball-statistics/foosball-videos/Westermann Bade VS Hoffmann Spredeman [1sBcrUBZxlY].mp4"
video_path = "/home/freystec/foosball-statistics/foosball-videos/SideKick Harburg - SiMo Doppel  05.08.2025_.mp4"
annotated_video_path = "output_video.avi" # Still used as an intermediate for moviepy
final_output_video_path = "SiMo_yolov11n_imgsz_640_long.mp4"
json_path = final_output_video_path.replace(".mp4", ".json")
temp_dir = "old/temp_predictions"

# Create the temporary directory for saving intermediate predictions
if not os.path.exists(temp_dir):
    os.makedirs(temp_dir)

# Load the YOLOv8 model
model = YOLO(model_path)

# --- Define Crop Parameters ---
# Adjust these values based on your video to focus on the foosball table
crop_left = 300  # pixels to cut from the left 350
crop_right = 300 # pixels to cut from the right 350
crop_top = 0     # pixels to cut from the top
crop_bottom = 0  # pixels to cut from the bottom

# --- Supervision Setup ---
video_info = sv.VideoInfo.from_video_path(video_path)
frame_generator = sv.get_video_frames_generator(video_path)

# Get original video dimensions from VideoInfo
width, height = video_info.width, video_info.height
total_frames = video_info.total_frames
fps = video_info.fps

# Calculate the dimensions of the cropped region for prediction
cropped_width = width - crop_left - crop_right
cropped_height = height - crop_top - crop_bottom

# Initialize frame counter (tqdm will handle this implicitly for iteration, but we keep it for data storage)
frame_count = 0

# List to store all predictions
predictions = []

# Start time for processing
start_time = time.time()

# Initialize VideoSink for writing the annotated video
# The VideoSink will use the original video_info, maintaining the original dimensions
with sv.VideoSink(annotated_video_path, video_info=video_info) as sink:
    # Loop through the video frames using the generator with tqdm for progress
    for frame_idx, frame in enumerate(tqdm(frame_generator, total=video_info.total_frames, desc="Processing Video")):
        # --- Crop the frame for prediction ---
        cropped_frame = frame[crop_top : height - crop_bottom, crop_left : width - crop_right]

        # Suppress YOLOv8 verbose output during inference
        with contextlib.redirect_stdout(open(os.devnull, 'w')):
            results = model(cropped_frame, verbose=False) # Perform inference on the cropped frame

        # Extract additional information
        inference_speed = results[0].speed

        # Create a copy of the original frame to draw annotations on
        annotated_frame = frame.copy()

        # Initialize the frame's prediction data
        frame_data = {
            "frame_count": frame_idx, # Use frame_idx from enumerate
            "inference_speed": inference_speed,
            "boxes": []
        }

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
            color = (0, 255, 0) # Green for bounding boxes
            thickness = 2
            cv2.rectangle(annotated_frame, (x1_original, y1_original), (x2_original, y2_original), color, thickness)
            
            label = f"Class {cls}: {conf:.2f}"
            cv2.putText(annotated_frame, label, (x1_original, y1_original - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # Write the annotated frame (with original dimensions) to the output video
        sink.write_frame(annotated_frame)

        # Append the frame's prediction data to the overall predictions
        predictions.append(frame_data)

        # Remove the manual progress update as tqdm handles it
        # if frame_count == 5 or frame_count % 100 == 0:
        #     elapsed_time = time.time() - start_time
        #     frames_processed = frame_count
        #     total_time_estimate = (elapsed_time / frames_processed) * total_frames
        #     remaining_time = (total_time_estimate - elapsed_time) / 60  # Convert to minutes
        #     print(f"Processed {frames_processed}/{total_frames} frames. "
        #           f"Estimated time remaining: {remaining_time:.2f} minutes.")


# Save predictions to a JSON file
with open(json_path, "w") as f:
    json.dump(predictions, f, indent=4)

# Merge video with original audio (MoviePy is still good for this)
processed_video = VideoFileClip(annotated_video_path)
original_video = VideoFileClip(video_path)

# Set the audio of the processed video to be the audio of the original video
processed_video = processed_video.with_audio(original_video.audio)

# Save the result
processed_video.write_videofile(final_output_video_path, codec='libx264', audio_codec='aac')

print(f"Predictions saved to {json_path}")
print(f"Video processing completed in {(time.time() - start_time) / 60:.2f} minutes.")