import json
import os
import time
import contextlib
import sys
from collections import defaultdict

import cv2
from ultralytics import YOLO
from moviepy.editor import VideoFileClip

# Define paths
model_path = r"C:\Users\chris\foosball-statistics\src\training_and_inference_scripts\runs\detect\train11\weights\yolov8m_imgsz_1280_with_topview_Leonhart_best.pt"
video_path = r"D:\Difficult Sequences\Bonzini_Beispiel.mp4"
annotated_video_path = "../../output_video.avi"
final_output_video_path = "Bonzini_Beispiel_imgsz1280_yolov8m_detection.mp4"
json_path = final_output_video_path.replace(".mp4", ".json")
temp_dir = "../old/temp_predictions"

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

# Initialize video writer
fourcc = cv2.VideoWriter_fourcc(*'XVID')
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
        # Suppress YOLOv8 verbose output during inference
        with contextlib.redirect_stdout(open(os.devnull, 'w')):
            results = model(frame)

        # Extract additional information
        inference_speed = results[0].speed

        # Visualize the results on the frame (annotate)
        annotated_frame = results[0].plot(labels=False, conf=True)

        # Write the annotated frame to the output video
        out.write(annotated_frame)

        # Initialize the frame's prediction data
        frame_data = {
            "frame_count": frame_count,
            "inference_speed": inference_speed,
            "boxes": []
        }

        # Extract predictions
        for result in results[0].boxes:
            bbox = result.xywhn[0].cpu().numpy().tolist()  # Normalized bounding box (xywh)
            conf = float(result.conf[0].cpu().numpy())  # Confidence
            cls = int(result.cls[0].cpu().numpy())  # Class

            # Append the box data to the frame's prediction data
            frame_data["boxes"].append({
                "bbox": bbox,
                "conf": conf,
                "cls": cls
            })

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
