from collections import defaultdict
import cv2
import numpy as np
from ultralytics import YOLO
from moviepy.editor import VideoFileClip
import time
import json
import pickle

# Define paths
model_path = r"C:\Users\chris\Foosball Detector\runs\detect\train23\weights\best.pt"
video_path = r"C:\Users\chris\Videos\vegas-thomas-haas-sarah-klabunde-vs-brandon-moreland-sullivan-rue-gz4kj_cO5Iqjuc.mp4"
final_output_video_path = "Small_Test.mp4"
# video_path = r"C:\Users\chris\Videos\vegas-thomas-haas-sarah-klabunde-vs-brandon-moreland-sullivan-rue-gz4kj_cO5Iqjuc.mp4"
# final_output_video_path = "testaarlardfdfdfdf.mp4"
annotated_video_path = "output_video.avi"
json_path = final_output_video_path.replace(".mp4", ".json")

#Load the YOLOv8 model
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

# Store the track history
track_history = defaultdict(lambda: [])

# Define constants
MAX_FRAMES_ABSENT = 10000
current_ball_id = None
next_ball_id = 10000
frames_ball_absent = 0
ball_class_index = None
current_ball_location_normalized = np.zeros(4)

# Initialize frame counter
frame_count = 0

# List to store all predictions
predictions = []

# Start time for processing
start_time = time.time()


while cap.isOpened():
    # Read a frame from the video
    success, frame = cap.read()

    if success:
        # Run YOLOv8 tracking on the frame, persisting tracks between frames
        results = model.track(frame, persist=True)

        # Extract additional information
        inference_speed = results[0].speed

        # Find the class index for "ball"
        if ball_class_index is None:

            # Get the class names for all detections
            all_class_names = results[0].names

            for index, name in all_class_names.items():
                if name == "ball":  # "ball (slow)"
                    ball_class_index = index
                    break
            # Check if ball_class_index is found
            if ball_class_index is None:
                print("Class 'ball' not found in the model classes.")
                exit()

        # Get the boxes, class labels (cls), and track IDs
        boxes = results[0].boxes.xywh.cpu()
        boxes_normalized = results[0].boxes.xywhn.cpu().numpy()
        labels = results[0].boxes.cls.cpu().tolist()  # Use 'cls' instead of 'labels'
        track_ids = results[0].boxes.id.int().cpu().tolist() if results[0].boxes.id is not None else []

        # Filter out the detections that are not of class "ball"
        valid_indices = [i for i, label in enumerate(labels) if label == ball_class_index]
        if track_ids:  # Check if track_ids is not empty
            boxes = [boxes[i] for i in valid_indices]
            boxes_normalized = [boxes_normalized[i] for i in valid_indices]
            track_ids = [track_ids[i] for i in valid_indices]
        else:
            boxes = []
            track_ids = []

        if len(boxes) == 0:
            frames_ball_absent += 1
            if frames_ball_absent > MAX_FRAMES_ABSENT:
                current_ball_id = None
        else:
            if current_ball_id is None:
                current_ball_id = next_ball_id
                next_ball_id += 1
            current_ball_location = boxes[0][:2]  # Use x, y from the box for consistency
            current_ball_location_normalized = boxes_normalized[0]
            track_history[current_ball_id].append(current_ball_location)
            frames_ball_absent = 0

        # Visualize the results on the frame
        annotated_frame = results[0].plot(labels=False, conf=False)

        # Plot the tracks
        for track_id, track in track_history.items():
            if len(track) > 0:
                # We are taking the most recent position for the ball.
                x, y = track[-1]
                if len(track) > 50:  # retain tracks for the last 20 frames
                    track.pop(0)
                # Draw the tracking lines
                points = np.hstack(track).astype(np.int32).reshape((-1, 1, 2))
                # cv2.polylines(annotated_frame, [points], isClosed=False, color=(230, 230, 230), thickness=10)
                cv2.polylines(annotated_frame, [points], isClosed=False, color=(0, 255, 255), thickness=4)

        # # Display the annotated frame
        # cv2.imshow("YOLOv8 Tracking", annotated_frame)

        out.write(annotated_frame)

        # Initialize the frame's prediction data
        frame_data = {
            "frame_count": frame_count,
            "inference_speed": inference_speed,
            "boxes": [],
            "current_ball_location": current_ball_location_normalized
        }

        # Extract predictions
        for result in results[0].boxes:
            bbox = result.xywhn[0].cpu().numpy().tolist()  # Normalized bounding box (xywh)
            conf = float(result.conf[0].cpu().numpy())  # Confidence
            cls = int(result.cls[0].cpu().numpy()) # Class
            ids = int(result.id[0].cpu().numpy()) if result.id is not None else -1

            # Append the box data to the frame's prediction data
            frame_data["boxes"].append({
                "bbox": bbox,
                "conf": conf,
                "cls": cls,
                "id": ids
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

# Release the video capture object and close the display window
cap.release()
out.release()
cv2.destroyAllWindows()

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