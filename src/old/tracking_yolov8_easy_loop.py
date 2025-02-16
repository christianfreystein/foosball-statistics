from collections import defaultdict
import cv2
import numpy as np
from ultralytics import YOLO
from moviepy.editor import VideoFileClip
import time
import json
import os

# Define paths and video list
model_path = r"C:\Users\chris\foosball-statistics\runs\detect\train5\weights\best.pt"
video_paths = [
    r"D:\Foosball Detector\videos\21.05.22 Stadtmeisterschaft OD Profi Finale Struth⧸Bechtel - Janßen⧸Stockmanns (online-video-cutter.com)_long.mp4",
    r"D:\Foosball Detector\videos\Bundesliga_2024_Bamberg_Kicker_Crew_Bonn.mp4",
    r"C:\Users\chris\Videos\Arnsberg 2011 - Amateur Doppel Finale  [2⧸2].mp4",
    r"C:\Users\chris\Videos\Tablesoccer.TV - Struth⧸Uhlemann vs Heinrich⧸Wahle.mp4",
    r"C:\Users\chris\Videos\Finals： Open Singles - ITSF World Series by Leonhart 2019 ｜ Tablesoccer.TV.mp4",
    r"C:\Users\chris\Videos\Tablesoccer.TV - Struth⧸Uhlemann vs Heinrich⧸Wahle.mp4",
    r"C:\Users\chris\Videos\Vegas  ｜  Thomas Haas & Sarah Klabunde vs Brandon Moreland & Sullivan Rue [Gz4kJPnpHXg].mp4"
]

# Load the YOLOv8 model
model = YOLO(model_path)

for video_path in video_paths:
    # Automatically create output file names based on the video name
    base_name = os.path.basename(video_path).replace(".mp4", "")
    final_output_video_path = f"{base_name}_processed.mp4"
    annotated_video_path = f"{base_name}_annotated.avi"
    json_path = f"{base_name}.json"

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
                all_class_names = results[0].names
                for index, name in all_class_names.items():
                    if name == "ball":
                        ball_class_index = index
                        break

            if ball_class_index is None:
                print(f"Class 'ball' not found in {video_path}.")
                break

            # Get the boxes, class labels (cls), and track IDs
            boxes = results[0].boxes.xywh.cpu()
            boxes_normalized = results[0].boxes.xywhn.cpu().numpy()
            labels = results[0].boxes.cls.cpu().tolist()
            track_ids = results[0].boxes.id.int().cpu().tolist() if results[0].boxes.id is not None else []

            valid_indices = [i for i, label in enumerate(labels) if label == ball_class_index]
            if track_ids:
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
                current_ball_location = boxes[0][:2]
                current_ball_location_normalized = boxes_normalized[0]
                track_history[current_ball_id].append(current_ball_location)
                frames_ball_absent = 0

            annotated_frame = results[0].plot(labels=False, conf=False)

            for track_id, track in track_history.items():
                if len(track) > 0:
                    x, y = track[-1]
                    if len(track) > 10:
                        track.pop(0)
                    points = np.hstack(track).astype(np.int32).reshape((-1, 1, 2))
                    cv2.polylines(annotated_frame, [points], isClosed=False, color=(0, 255, 255), thickness=4)

            out.write(annotated_frame)

            frame_data = {
                "frame_count": frame_count,
                "inference_speed": inference_speed,
                "boxes": [],
                "current_ball_location": current_ball_location_normalized.tolist()
            }

            for result in results[0].boxes:
                bbox = result.xywhn[0].cpu().numpy().tolist()
                conf = float(result.conf[0].cpu().numpy())
                cls = int(result.cls[0].cpu().numpy())
                ids = int(result.id[0].cpu().numpy()) if result.id is not None else -1

                frame_data["boxes"].append({
                    "bbox": bbox,
                    "conf": conf,
                    "cls": cls,
                    "id": ids
                })

            predictions.append(frame_data)

            frame_count += 1

            if frame_count == 5 or frame_count % 100 == 0:
                elapsed_time = time.time() - start_time
                frames_processed = frame_count
                total_time_estimate = (elapsed_time / frames_processed) * total_frames
                remaining_time = (total_time_estimate - elapsed_time) / 60
                print(f"Processed {frames_processed}/{total_frames} frames from {video_path}. "
                      f"Estimated time remaining: {remaining_time:.2f} minutes.")

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        else:
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()

    with open(json_path, "w") as f:
        json.dump(predictions, f, indent=4)

    # Merge video with original audio
    processed_video = VideoFileClip(annotated_video_path)
    original_video = VideoFileClip(video_path)
    processed_video = processed_video.set_audio(original_video.audio)
    processed_video.write_videofile(final_output_video_path, codec='libx264', audio_codec='aac')

    print(f"Predictions saved to {json_path}")
    print(f"Video processing completed for {video_path} in {(time.time() - start_time) / 60:.2f} minutes.")
