import json
import cv2
import numpy as np
from ultralytics.trackers.bot_sort import BOTSORT

# Define paths directly in the code
VIDEO_PATH = r"C:\Users\chris\foosball-statistics\src\training_and_inference_scripts\Leonhart_Clip_Topview_yolov8m_1280_imgsz_detection.mp4"
DETECTIONS_PATH = r"C:\Users\chris\foosball-statistics\src\training_and_inference_scripts\Leonhart_Clip_Topview_yolov8m_1280_imgsz_detection.json"

# Define tracking parameters directly
TRACKING_PARAMS = {
    "proximity_thresh": 0.5,
    "appearance_thresh": 0.3,
    "gmc_method": "none",  # Change "default" to "orb" or "none"
    "with_reid": False,
    "fuse_score": False,
    "track_buffer": 60,
}

# Convert bbox from (cx, cy, w, h) → (x, y, w, h)
def convert_bbox(img_width, img_height, bbox):
    cx, cy, w, h = bbox  # Center coordinates with width and height
    x = (cx - w / 2) * img_width
    y = (cy - h / 2) * img_height
    w = w * img_width
    h = h * img_height
    return [x, y, w, h]  # Return (x, y, w, h)


# Load detection results
with open(DETECTIONS_PATH, "r") as f:
    detections = json.load(f)

# Initialize BOTSORT with direct parameters
class Args:
    def __init__(self, params):
        self.proximity_thresh = params["proximity_thresh"]
        self.appearance_thresh = params["appearance_thresh"]
        self.gmc_method = params["gmc_method"]
        self.with_reid = params["with_reid"]
        self.fuse_score = params["fuse_score"]
        self.track_buffer = params["track_buffer"]  # Fix: Added this parameter

args = Args(TRACKING_PARAMS)
tracker = BOTSORT(args, frame_rate=60)

# Open video file
cap = cv2.VideoCapture(VIDEO_PATH)

frame_idx = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    img_height, img_width, _ = frame.shape  # Get image dimensions

    # Get detections for the current frame
    frame_data = next((item for item in detections if item["frame_count"] == frame_idx), None)
    dets = []
    scores = []
    classes = []
    idx = 0  # Start from 0 and increment for each detection

    for detection in frame_data["boxes"]:  # Loop through all detections in a frame
        bbox = detection["bbox"]  # (cx, cy, w, h)
        conf = detection["conf"]
        cls = detection["cls"]

        # Convert bbox format (center-based to top-left-based)
        x = bbox[0]  # Assuming bbox format is already in (cx, cy, w, h)
        y = bbox[1]
        w = bbox[2]
        h = bbox[3]

        # Append detection with confidence, class, and idx
        dets.append(np.array([x, y, w, h, idx]))  # Now includes idx
        scores.append(conf)  # Scores are used separately
        classes.append(cls)  # Class labels

        idx += 1  # Increment idx for the next detection

    # Initialize tracking
    tracks = tracker.init_track(dets, scores, classes, img=frame)

    # Predict & update tracks
    tracker.multi_predict(tracks)

    # Draw tracked objects
    for track in tracks:
        x1, y1, x2, y2 = track.tlwh
        track_id = track.track_id
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        cv2.putText(frame, f"ID {track_id}", (int(x1), int(y1)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Show frame
    cv2.imshow("Tracking", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

    frame_idx += 1

cap.release()
cv2.destroyAllWindows()
