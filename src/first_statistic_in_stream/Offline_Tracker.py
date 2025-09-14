import cv2
import json
import numpy as np
from collections import deque
from tqdm import tqdm
import os


# --- Re-including necessary classes from foosball-tracker-offline ---
# NOTE: These classes are kept as-is for the tracking logic.
# They are not part of the new functions but are used by them.
# The code below will include the full classes from your original script.
class MockKalmanFilter:
    def __init__(self, initial_position, process_noise_scale_q=1.0, measurement_noise_scale_r=1.0):
        self.state = np.array([initial_position[0], initial_position[1], 0.0, 0.0])
        self.P = np.eye(4) * 100.0
        self.F = np.array([
            [1, 0, 1, 0],
            [0, 1, 0, 1],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])
        self.H = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0]
        ])
        self.Q = np.eye(4) * process_noise_scale_q
        self.Q[0, 0] = self.Q[1, 1] = process_noise_scale_q * 0.5
        self.Q[2, 2] = self.Q[3, 3] = process_noise_scale_q * 1.0
        self.R = np.eye(2) * measurement_noise_scale_r

    def predict(self):
        self.state = np.dot(self.F, self.state)
        self.P = np.dot(np.dot(self.F, self.P), self.F.T) + self.Q
        return self.state[0:2]

    def update(self, measurement):
        S = np.dot(np.dot(self.H, self.P), self.H.T) + self.R
        K = np.dot(np.dot(self.P, self.H.T), np.linalg.inv(S))
        y = measurement - np.dot(self.H, self.state)
        self.state = self.state + np.dot(K, y)
        I = np.eye(self.state.shape[0])
        self.P = np.dot((I - np.dot(K, self.H)), self.P)
        return self.state[0:2]


class Detection:
    def __init__(self, frame_idx, bbox, confidence):
        self.frame_idx = frame_idx
        self.bbox = bbox
        self.confidence = confidence
        self.center = ((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2)


class Track:
    _next_id = 0

    def __init__(self, initial_detection, process_noise_scale_q=1.0, measurement_noise_scale_r=1.0):
        self.track_id = Track._next_id
        Track._next_id += 1
        self.kalman_filter = MockKalmanFilter(initial_detection.center, process_noise_scale_q,
                                              measurement_noise_scale_r)
        self.detections = [initial_detection]
        self.last_detection_frame = initial_detection.frame_idx
        self.last_known_position = initial_detection.center
        self.state = 'active'
        self.frames_since_last_detection = 0

    def add_detection(self, detection):
        self.detections.append(detection)
        self.kalman_filter.update(np.array(detection.center))
        self.last_detection_frame = detection.frame_idx
        self.last_known_position = detection.center
        self.frames_since_last_detection = 0
        self.state = 'active'

    def mark_lost(self):
        self.frames_since_last_detection += 1
        self.kalman_filter.predict()
        self.state = 'lost'

    def get_predicted_position(self):
        return self.kalman_filter.predict()


class OfflineFoosballTracker:
    def __init__(self, conf_threshold=0.6, max_lost_frames=30, min_init_detections=3,
                 reid_proximity_thresh=50, process_noise_scale_q=1.0, measurement_noise_scale_r=1.0,
                 max_frames_for_aggressive_reid=5, aggressive_reid_distance_factor=5.0):
        self.conf_threshold = conf_threshold
        self.max_lost_frames = max_lost_frames
        self.min_init_detections = min_init_detections
        self.reid_proximity_thresh = reid_proximity_thresh
        self.process_noise_scale_q = process_noise_scale_q
        self.measurement_noise_scale_r = measurement_noise_scale_r
        self.max_frames_for_aggressive_reid = max_frames_for_aggressive_reid
        self.aggressive_reid_distance_threshold = reid_proximity_thresh * aggressive_reid_distance_factor
        self.active_tracks = []
        self.terminated_tracks = []
        self.potential_new_tracks = deque()
        self.frame_to_tracked_detections_map = {}
        self.all_tracks_by_id = {}

    def euclidean_distance(self, p1, p2):
        return np.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)

    def process_video(self, all_detections_by_frame):
        num_frames = max(all_detections_by_frame.keys()) + 1 if all_detections_by_frame else 0
        for frame_idx in tqdm(range(num_frames), desc="Processing Frames for Tracking"):
            current_frame_raw_detections = all_detections_by_frame.get(frame_idx, [])
            current_frame_detections = [
                Detection(frame_idx, d['bbox'], d['confidence'])
                for d in current_frame_raw_detections if d['confidence'] >= self.conf_threshold
            ]
            self.frame_to_tracked_detections_map[frame_idx] = []
            matched_detections_indices = set()
            for track in self.active_tracks:
                if track.state == 'active':
                    predicted_pos = track.get_predicted_position()
                    min_dist = float('inf')
                    best_match_idx = -1
                    for i, det in enumerate(current_frame_detections):
                        if i in matched_detections_indices:
                            continue
                        dist = self.euclidean_distance(predicted_pos, det.center)
                        if dist < min_dist and dist < self.reid_proximity_thresh:
                            min_dist = dist
                            best_match_idx = i
                    if best_match_idx != -1:
                        track.add_detection(current_frame_detections[best_match_idx])
                        self.frame_to_tracked_detections_map[frame_idx].append(
                            (current_frame_detections[best_match_idx], track.track_id))
                        matched_detections_indices.add(best_match_idx)
            unmatched_detections = [
                det for i, det in enumerate(current_frame_detections)
                if i not in matched_detections_indices
            ]
            matched_unmatched_indices_aggressive = set()
            unmatched_detections.sort(key=lambda x: x.confidence, reverse=True)
            for track in self.active_tracks:
                if track.state == 'lost' and track.frames_since_last_detection <= self.max_frames_for_aggressive_reid:
                    min_dist_aggressive_reid = float('inf')
                    best_unmatched_det_idx = -1
                    for j, unmatched_det in enumerate(unmatched_detections):
                        if j in matched_unmatched_indices_aggressive:
                            continue
                        dist = self.euclidean_distance(track.last_known_position, unmatched_det.center)
                        if dist < min_dist_aggressive_reid and dist < self.aggressive_reid_distance_threshold:
                            min_dist_aggressive_reid = dist
                            best_unmatched_det_idx = j
                    if best_unmatched_det_idx != -1:
                        track.add_detection(unmatched_detections[best_unmatched_det_idx])
                        self.frame_to_tracked_detections_map[frame_idx].append(
                            (unmatched_detections[best_unmatched_det_idx], track.track_id))
                        matched_unmatched_indices_aggressive.add(best_unmatched_det_idx)
            unmatched_detections_after_reid = [
                det for j, det in enumerate(unmatched_detections)
                if j not in matched_unmatched_indices_aggressive
            ]
            tracks_to_remove = []
            for track in self.active_tracks:
                if track.last_detection_frame < frame_idx:
                    track.mark_lost()
                    if track.frames_since_last_detection > self.max_lost_frames:
                        track.state = 'terminated'
                        self.terminated_tracks.append(track)
                        tracks_to_remove.append(track)
            self.active_tracks = [t for t in self.active_tracks if t.state != 'terminated']
            for det in unmatched_detections_after_reid:
                self.potential_new_tracks.append(det)
            while self.potential_new_tracks and \
                    self.potential_new_tracks[0].frame_idx < frame_idx - (
                    self.min_init_detections + self.max_frames_for_aggressive_reid + 5):
                self.potential_new_tracks.popleft()
            newly_started_tracks_indices = []
            for i in range(len(self.potential_new_tracks)):
                current_det = self.potential_new_tracks[i]
                consecutive_count = 1
                temp_detections_for_init = [current_det]
                for j in range(i + 1, len(self.potential_new_tracks)):
                    next_det = self.potential_new_tracks[j]
                    if next_det.frame_idx == current_det.frame_idx + consecutive_count and \
                            self.euclidean_distance(current_det.center, next_det.center) < 150:
                        consecutive_count += 1
                        temp_detections_for_init.append(next_det)
                    else:
                        break
                if consecutive_count >= self.min_init_detections:
                    new_track = Track(temp_detections_for_init[0],
                                      self.process_noise_scale_q, self.measurement_noise_scale_r)
                    self.active_tracks.append(new_track)
                    self.all_tracks_by_id[new_track.track_id] = new_track
                    for k in range(1, len(temp_detections_for_init)):
                        new_track.add_detection(temp_detections_for_init[k])
                    for init_det in temp_detections_for_init:
                        if init_det.frame_idx == frame_idx:
                            self.frame_to_tracked_detections_map[frame_idx].append((init_det, new_track.track_id))
                    newly_started_tracks_indices.append(i)
            for idx in sorted(newly_started_tracks_indices, reverse=True):
                del self.potential_new_tracks[idx]
        return self.frame_to_tracked_detections_map


def save_tracked_data_to_json(tracked_results_by_frame, output_json_path, frame_width, frame_height):
    json_output_data = []
    for frame_idx in sorted(tracked_results_by_frame.keys()):
        frame_data = {
            "frame_count": frame_idx,
            "boxes": [],
            "current_ball_location": None,
            "ball_status": "Unknown"
        }
        best_ball_confidence = -1
        best_ball_bbox = None
        for det, track_id in tracked_results_by_frame[frame_idx]:
            x_min, y_min, x_max, y_max = det.bbox
            width = (x_max - x_min) / frame_width
            height = (y_max - y_min) / frame_height
            x_center = (x_min + x_max) / 2 / frame_width
            y_center = (y_min + y_max) / 2 / frame_height
            box_entry = {
                "bbox": [x_center, y_center, width, height],
                "conf": det.confidence,
                "cls": 0,
                "id": track_id
            }
            frame_data["boxes"].append(box_entry)
            if det.confidence > best_ball_confidence:
                best_ball_confidence = det.confidence
                best_ball_bbox = box_entry["bbox"]
        if best_ball_bbox:
            frame_data["current_ball_location"] = best_ball_bbox
        json_output_data.append(frame_data)
    print(f"Saving tracked data to {output_json_path}...")
    with open(output_json_path, 'w') as f:
        json.dump(json_output_data, f, indent=4)
    print("Tracked data saved successfully.")


# New functions below
# --------------------

def read_detections_from_json(detection_json_path: str) -> list:
    """
    Reads and loads detection data from a JSON file.

    Args:
        detection_json_path (str): The path to the input JSON file.

    Returns:
        list: The raw detection data as a list of dictionaries.
    """
    print("Loading detections from JSON...")
    with open(detection_json_path, 'r') as f:
        raw_detections_data = json.load(f)
    print(f"Loaded detections from {detection_json_path}.")
    return raw_detections_data


def run_tracking_on_detections(raw_detections_data: list, frame_width: int, frame_height: int,
                               tracker_params: dict) -> tuple:
    """
    Processes a list of raw detections to create object tracks.

    Args:
        raw_detections_data (list): A list of dictionaries, where each dict represents a frame's detections.
        frame_width (int): The width of the video frames in pixels.
        frame_height (int): The height of the video frames in pixels.
        tracker_params (dict): A dictionary of parameters for the OfflineFoosballTracker.

    Returns:
        tuple: A tuple containing:
            - dict: A dictionary of tracked detections by frame index.
            - OfflineFoosballTracker: The tracker object itself, useful for accessing all track data.
    """
    # Prepare detections for the tracker, converting normalized bbox to pixel coordinates
    # and structuring by frame_idx
    all_detections_by_frame = {}

    for frame_data in raw_detections_data:
        frame_idx = frame_data['frame_count']
        current_frame_detections = []
        for box_data in frame_data.get('boxes', []):
            if box_data.get('cls') == 0:  # Only interested in 'ball' detections (class 0)
                # Convert normalized [x_center, y_center, width, height] to [x_min, y_min, x_max, y_max] pixels
                x_center, y_center, width, height = box_data['bbox']

                x_min = int((x_center - width / 2) * frame_width)
                y_min = int((y_center - height / 2) * frame_height)
                x_max = int((x_center + width / 2) * frame_width)
                y_max = int((y_center + height / 2) * frame_height)

                # Clamp values to frame boundaries
                x_min = max(0, x_min)
                y_min = max(0, y_min)
                x_max = min(frame_width, x_max)
                y_max = min(frame_height, y_max)

                current_frame_detections.append({
                    'bbox': [x_min, y_min, x_max, y_max],
                    'confidence': box_data.get('conf', 0.0)
                })
        all_detections_by_frame[frame_idx] = current_frame_detections

    # Initialize and run the OfflineFoosballTracker with passed parameters
    tracker = OfflineFoosballTracker(**tracker_params)
    print("Running offline tracking...")
    tracked_results_by_frame = tracker.process_video(all_detections_by_frame)
    print("Tracking complete.")

    return tracked_results_by_frame, tracker


def visualize_tracking_on_video(video_path: str = None, frames: list = None, fps: float = None,
                                tracked_results_by_frame: dict = None, tracker=None,
                                trail_length_frames: int = 60, output_video_path: str = None):
    """
    Annotates a video with object tracking information (bounding boxes and trails).

    This function processes a video, adds bounding boxes, track IDs, and fading trails.
    It can accept either a video file path or a list of frames.

    Args:
        video_path (str, optional): Path to the input video file.
        frames (list, optional): A list of pre-loaded frames (NumPy arrays).
        fps (float, optional): The frames per second (FPS) of the video. Required if
                               providing 'frames' and no 'video_path'.
        tracked_results_by_frame (dict): A dictionary of tracked detections by frame index.
        tracker (OfflineFoosballTracker): The tracker object containing all track data.
        trail_length_frames (int): The number of frames for which to draw the tracking trail.
        output_video_path (str, optional): Path to save the annotated video. If None,
                                          the video will not be saved.

    Returns:
        list: A list of annotated frames (NumPy arrays).

    Raises:
        ValueError: If both video_path and frames are provided, or if neither is provided,
                    or if fps is not provided with frames.
        IOError: If the input video or output video writer cannot be opened.
    """
    if video_path is None and frames is None:
        raise ValueError("Either video_path or frames must be provided.")
    if video_path and frames:
        raise ValueError("Cannot provide both video_path and frames.")

    cap = None
    frame_source = None
    total_frames = 0
    frame_width = 0
    frame_height = 0
    video_fps = 0

    if video_path:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise IOError(f"Cannot open video file: {video_path}")
        frame_source = cap
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        video_fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    elif frames:
        if not frames:
            raise ValueError("Provided 'frames' list is empty.")
        if fps is None:
            raise ValueError("FPS must be provided when passing frames directly.")
        frame_source = frames
        total_frames = len(frames)
        video_fps = fps
        frame_height, frame_width, _ = frames[0].shape

    print(f"Video Info: {frame_width}x{frame_height} @ {video_fps:.2f} FPS, {total_frames} frames")

    out = None
    if output_video_path:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_video_path, fourcc, video_fps, (frame_width, frame_height))
        if not out.isOpened():
            raise IOError(f"Could not open video writer for: {output_video_path}")
        print(f"Writing annotated video to: {output_video_path}")

    annotated_frames = []
    track_color_palette = [
        (0, 255, 0), (255, 0, 0), (0, 0, 255), (0, 255, 255),
        (255, 255, 0), (255, 0, 255), (128, 0, 128), (0, 128, 128),
        (128, 128, 0), (0, 165, 255)
    ]
    track_colors = {}

    with tqdm(total=total_frames, desc="Annotating Video") as pbar:
        for current_frame_idx in range(total_frames):
            frame = None
            if video_path:
                ret, frame = frame_source.read()
                if not ret:
                    break
            else:
                frame = frames[current_frame_idx]

            # --- Draw Track Trails ---
            for track_id, track_obj in tracker.all_tracks_by_id.items():
                if track_obj.detections:
                    trail_points_with_frames = []
                    for det in track_obj.detections:
                        if det.frame_idx <= current_frame_idx and (
                                current_frame_idx - det.frame_idx) < trail_length_frames:
                            trail_points_with_frames.append((det.center, det.frame_idx))

                    trail_points_with_frames.sort(key=lambda x: x[1])

                    if len(trail_points_with_frames) > 1:
                        if track_id not in track_colors:
                            track_colors[track_id] = track_color_palette[track_id % len(track_color_palette)]

                        base_color = track_colors[track_id]

                        for i in range(1, len(trail_points_with_frames)):
                            p1_center, p1_frame_idx = trail_points_with_frames[i - 1]
                            p2_center, p2_frame_idx = trail_points_with_frames[i]
                            segment_age = current_frame_idx - p2_frame_idx
                            age_ratio = float(segment_age) / max(1, trail_length_frames - 1)
                            fade_factor = max(0.4, 1.0 - age_ratio)

                            faded_color = (
                                int(base_color[0] * fade_factor),
                                int(base_color[1] * fade_factor),
                                int(base_color[2] * fade_factor)
                            )
                            cv2.line(frame, (int(p1_center[0]), int(p1_center[1])),
                                     (int(p2_center[0]), int(p2_center[1])),
                                     faded_color, 4)

                            # --- Draw Current Frame Detections (Bounding Box, ID, Confidence) ---
            tracked_dets_for_frame = tracked_results_by_frame.get(current_frame_idx, [])
            for det, track_id in tracked_dets_for_frame:
                x1, y1, x2, y2 = det.bbox
                conf = det.confidence

                if track_id not in track_colors:
                    track_colors[track_id] = track_color_palette[track_id % len(track_color_palette)]
                current_track_color = track_colors[track_id]

                thickness = 2
                cv2.rectangle(frame, (x1, y1), (x2, y2), current_track_color, thickness)

                label = f"ID: {track_id} | Conf: {conf:.2f}"
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.7
                font_thickness = 2
                text_size = cv2.getTextSize(label, font, font_scale, font_thickness)[0]
                text_x = x1
                text_y = y1 - 10 if y1 - 10 > 10 else y1 + text_size[1] + 10

                cv2.rectangle(frame, (text_x, text_y - text_size[1] - 5),
                              (text_x + text_size[0] + 5, text_y + 5), (0, 0, 0), -1)
                cv2.putText(frame, label, (text_x, text_y), font, font_scale, (255, 255, 255), font_thickness,
                            cv2.LINE_AA)

            annotated_frames.append(frame)

            if out:
                out.write(frame)

            pbar.update(1)

    if cap:
        cap.release()
    if out:
        out.release()
        print("Video annotation complete.")
        print(f"Output video saved to: {output_video_path}")

    return annotated_frames


if __name__ == "__main__":
    # --- Configuration ---
    input_video_path = r"C:\Users\chris\foosball-statistics\foosball-videos\SideKick Harburg - SiMo Doppel  05.08.2025.mp4"
    detection_json_path = r"C:\Users\chris\foosball-statistics\SiMo_yolov11n_imgsz_640.json"
    output_video_path = r"SiMo_yolov11n_imgsz_640_tracked.mp4"
    output_tracked_json_path = r"SiMo_yolov11n_imgsz_640_short_tracked.json"

    # Tracker parameters (unchanged)
    tracker_parameters = {
        "conf_threshold": 0.7,
        "max_lost_frames": 60,
        "min_init_detections": 6,
        "reid_proximity_thresh": 1000,
        "process_noise_scale_q": 100.0,
        "measurement_noise_scale_r": 0.1,
        "max_frames_for_aggressive_reid": 30,
        "aggressive_reid_distance_factor": 5.0
    }
    desired_trail_length_frames = 60

    try:
        # Get video properties for coordinate conversion and video writing
        cap_temp = cv2.VideoCapture(input_video_path)
        if not cap_temp.isOpened():
            raise IOError(f"Cannot open video file: {input_video_path}")
        frame_width = int(cap_temp.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap_temp.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap_temp.release()

        # Step 1: Read the JSON file
        # NOTE: The functions 'read_detections_from_json', 'run_tracking_on_detections', and 'save_tracked_data_to_json' are not provided in this file.
        # They are assumed to exist elsewhere in the program.
        detections_data = read_detections_from_json(detection_json_path)

        # Step 2: Run the tracking logic on the data
        tracked_results, tracker_obj = run_tracking_on_detections(detections_data, frame_width, frame_height,
                                                                  tracker_parameters)

        # Optional: Save the tracked data to a new JSON file
        if output_tracked_json_path:
            save_tracked_data_to_json(tracked_results, output_tracked_json_path, frame_width, frame_height)

        # Step 3: Visualize the tracked information on the video
        # The function now returns the list of annotated frames, which you can use for further processing.
        # # This first call will save the video because output_video_path is provided.
        # print("\nRunning visualization and saving the video...")
        # annotated_frames_saved = visualize_tracking_on_video(video_path=input_video_path,
        #                                                      tracked_results_by_frame=tracked_results,
        #                                                      tracker=tracker_obj,
        #                                                      trail_length_frames=desired_trail_length_frames,
        #                                                      output_video_path=output_video_path)
        # print(f"Number of annotated frames returned from the first call: {len(annotated_frames_saved)}")

        # This second call will NOT save the video because output_video_path is None.
        print("\nRunning visualization without saving the video...")
        annotated_frames_not_saved = visualize_tracking_on_video(video_path=input_video_path,
                                                                 tracked_results_by_frame=tracked_results,
                                                                 tracker=tracker_obj,
                                                                 trail_length_frames=desired_trail_length_frames,
                                                                 output_video_path=output_video_path)
        print(f"Number of annotated frames returned from the second call: {len(annotated_frames_not_saved)}")

    except Exception as e:
        print(f"An error occurred: {e}")
        print("Please ensure your video path and detection JSON path are correct and accessible.")
        print("Also, make sure 'opencv-python' and 'tqdm' are installed (`pip install opencv-python tqdm`).")