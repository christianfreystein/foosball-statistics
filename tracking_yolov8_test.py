from collections import defaultdict
import cv2
import numpy as np
from ultralytics import YOLO
from moviepy.editor import VideoFileClip

# Load the YOLOv8 model
model = YOLO(r"C:\Users\chris\Foosball Detector\runs\detect\train20\weights\best.pt")

# Open the video file
video_path = r"C:\Users\chris\Videos\vegas-thomas-haas-sarah-klabunde-vs-brandon-moreland-sullivan-rue-gz4kj_cO5Iqjuc.mp4"
cap = cv2.VideoCapture(video_path)

# Get properties of the original video
fps = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

# Initialize video writer
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output.avi', fourcc, fps, (width, height))

# Store the track history
track_history = defaultdict(lambda: [])

# Define constants
MAX_FRAMES_ABSENT = 50
current_ball_id = None
next_ball_id = 10000
frames_ball_absent = 0
ball_class_index = None

while cap.isOpened():
    # Read a frame from the video
    success, frame = cap.read()

    if success:
        # Run YOLOv8 tracking on the frame, persisting tracks between frames
        results = model.track(frame, persist=True)

        # Get the boxes, class labels (cls), and track IDs
        if results is not None and len(results) > 0 and results[0] is not None and results[0].boxes is not None:
            # Extract bounding box coordinates (x, y, w, h), class labels, and track IDs
            boxes = results[0].boxes.xywh.cpu()
            labels = results[0].boxes.cls.cpu().tolist()  # Use 'cls' instead of 'labels'

            if results[0].boxes.id is not None:
                track_ids = results[0].boxes.id.int().cpu().tolist()
            else:
                # If track IDs are not available, set them to None
                track_ids = [None] * len(boxes)

            # Filter out the detections that are not of class "ball"
            valid_indices = [i for i, label in enumerate(labels) if label == ball_class_index]
            boxes = [boxes[i] for i in valid_indices]
            track_ids = [track_ids[i] for i in valid_indices]

            if len(boxes) == 0:
                frames_ball_absent += 1
                if frames_ball_absent > MAX_FRAMES_ABSENT:
                    current_ball_id = None
            else:
                if current_ball_id is None:
                    current_ball_id = next_ball_id
                    next_ball_id += 1
                current_ball_location = boxes[0][:2]  # Use x, y from the box for consistency
                track_history[current_ball_id].append(current_ball_location)
                frames_ball_absent = 0
        else:
            frames_ball_absent += 1
            if frames_ball_absent > MAX_FRAMES_ABSENT:
                current_ball_id = None

        # Visualize the results on the frame
        annotated_frame = results[0].plot(labels=False, conf=False)

        # Plot the tracks
        for track_id, track in track_history.items():
            if len(track) > 0:
                x, y = track[-1]
                if len(track) > 10000:  # retain tracks for the last 20 frames
                    track.pop(0)
                points = np.hstack(track).astype(np.int32).reshape((-1, 1, 2))
                cv2.polylines(annotated_frame, [points], isClosed=False, color=(0, 255, 255), thickness=4)

        out.write(annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        break

cap.release()
out.release()
cv2.destroyAllWindows()

processed_video = VideoFileClip("output.avi")
original_video = VideoFileClip(video_path)
processed_video = processed_video.set_audio(original_video.audio)
processed_video.write_videofile("testerla.mp4", codec='libx264', audio_codec='aac')
