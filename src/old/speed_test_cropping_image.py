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
model_path = r"/runs/detect/train5/weights/best_run5.pt"
video_path = r"C:\Users\chris\foosball-statistics\Leonhart_clip.mp4"
annotated_video_path = "../../output_video.avi"
final_output_video_path = "test.mp4"
json_path = final_output_video_path.replace(".mp4", ".json")
temp_dir = "temp_predictions"

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
        # Measure cropping time
        crop_start = time.time()
        cropped_frame = frame[50:height - 50, 50:width - 50]  # Example crop (removes 50px from all sides)
        crop_time = time.time() - crop_start

        # Measure inference time
        inference_start = time.time()
        with contextlib.redirect_stdout(open(os.devnull, 'w')):
            results = model(cropped_frame)
        inference_time = time.time() - inference_start

        # Extract additional information
        inference_speed = results[0].speed

        # Measure annotation writing time
        write_start = time.time()
        annotated_frame = results[0].plot(labels=False, conf=False)
        out.write(annotated_frame)
        write_time = time.time() - write_start

        # Measure data extraction time
        extraction_start = time.time()
        frame_data = {
            "frame_count": frame_count,
            "crop_time": crop_time,
            "inference_time": inference_time,
            "write_time": write_time,
            "inference_speed": inference_speed,
            "boxes": []
        }

        for result in results[0].boxes:
            bbox = result.xywhn[0].cpu().numpy().tolist()
            conf = float(result.conf[0].cpu().numpy())
            cls = int(result.cls[0].cpu().numpy())

            frame_data["boxes"].append({
                "bbox": bbox,
                "conf": conf,
                "cls": cls
            })
        predictions.append(frame_data)
        extraction_time = time.time() - extraction_start

        # Print all measured times per iteration
        print(f"Frame {frame_count}:\n"
              f"  Crop Time: {crop_time:.4f} sec\n"
              f"  Inference Time: {inference_time:.4f} sec\n"
              f"  Write Time: {write_time:.4f} sec\n"
              f"  Extraction Time: {extraction_time:.4f} sec")

        # Increment the frame counter
        frame_count += 1

        # Provide progress update
        if frame_count == 5 or frame_count % 100 == 0:
            elapsed_time = time.time() - start_time
            frames_processed = frame_count
            total_time_estimate = (elapsed_time / frames_processed) * total_frames
            remaining_time = (total_time_estimate - elapsed_time) / 60
            print(f"Processed {frames_processed}/{total_frames} frames. "
                  f"Estimated time remaining: {remaining_time:.2f} minutes.")

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
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
processed_video = processed_video.set_audio(original_video.audio)
processed_video.write_videofile(final_output_video_path, codec='libx264', audio_codec='aac')

print(f"Predictions saved to {json_path}")
print(f"Video processing completed in {(time.time() - start_time) / 60:.2f} minutes.")
