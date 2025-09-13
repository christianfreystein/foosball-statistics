import cv2
import json
import numpy as np
from collections import deque
from tqdm import tqdm # For progress bar

# --- Re-including necessary classes from foosball-tracker-offline ---
# This ensures the visualization script is self-contained.

class MockKalmanFilter:
    def __init__(self, initial_position, process_noise_scale_q=1.0, measurement_noise_scale_r=1.0):
        # State: [x, y, vx, vy]
        self.state = np.array([initial_position[0], initial_position[1], 0.0, 0.0])
        # Covariance matrix (uncertainty)
        self.P = np.eye(4) * 100.0 # High initial uncertainty

        # State transition matrix (simple constant velocity model)
        # dt is assumed to be 1 (frame to frame) for simplicity here
        self.F = np.array([
            [1, 0, 1, 0], # x = x_prev + vx
            [0, 1, 0, 1], # y = y_prev + vy
            [0, 0, 1, 0], # vx = vx_prev
            [0, 0, 0, 1]  # vy = vy_prev
        ])

        # Measurement matrix (we only measure position x, y)
        self.H = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0]
        ])

        # Process Noise Covariance Q (Crucial for foosball: high to adapt to sudden changes)
        # Represents uncertainty in the model. A higher Q allows the filter to
        # 'trust' new measurements more and adapt faster to abrupt changes.
        # Now adjustable from outside
        self.Q = np.eye(4) * process_noise_scale_q
        self.Q[0, 0] = self.Q[1, 1] = process_noise_scale_q * 0.5 # Position noise component
        self.Q[2, 2] = self.Q[3, 3] = process_noise_scale_q * 1.0 # Velocity noise component

        # Measurement Noise Covariance R (Low as bounding box quality is good)
        # Represents uncertainty in the measurements. Now adjustable from outside
        self.R = np.eye(2) * measurement_noise_scale_r

    def predict(self):
        # Predict next state
        self.state = np.dot(self.F, self.state)
        # Predict next covariance
        self.P = np.dot(np.dot(self.F, self.P), self.F.T) + self.Q
        return self.state[0:2] # Return predicted position

    def update(self, measurement):
        # Calculate Kalman Gain
        S = np.dot(np.dot(self.H, self.P), self.H.T) + self.R
        K = np.dot(np.dot(self.P, self.H.T), np.linalg.inv(S))

        # Update state with measurement
        y = measurement - np.dot(self.H, self.state)
        self.state = self.state + np.dot(K, y)

        # Update covariance
        I = np.eye(self.state.shape[0])
        self.P = np.dot((I - np.dot(K, self.H)), self.P)

        return self.state[0:2] # Return updated position

class Detection:
    def __init__(self, frame_idx, bbox, confidence):
        # bbox stores [x_min, y_min, x_max, y_max] in pixel coordinates
        self.frame_idx = frame_idx
        self.bbox = bbox
        self.confidence = confidence
        self.center = ((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2)

class Track:
    _next_id = 0 # Class-level unique ID for tracks

    def __init__(self, initial_detection, process_noise_scale_q=1.0, measurement_noise_scale_r=1.0):
        self.track_id = Track._next_id
        Track._next_id += 1

        # Pass Kalman filter parameters during track initialization
        self.kalman_filter = MockKalmanFilter(initial_detection.center, process_noise_scale_q, measurement_noise_scale_r)
        self.detections = [initial_detection] # Store all detections associated with this track
        self.last_detection_frame = initial_detection.frame_idx
        self.last_known_position = initial_detection.center
        self.state = 'active' # 'active', 'lost', 'terminated'
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
        # Predict the state even when lost, but don't use it for output unless re-identified
        self.kalman_filter.predict()
        self.state = 'lost'

    def get_predicted_position(self):
        return self.kalman_filter.predict() # This prediction is internal for association

class OfflineFoosballTracker:
    def __init__(self, conf_threshold=0.6, max_lost_frames=30, min_init_detections=3, 
                     reid_proximity_thresh=50, process_noise_scale_q=1.0, measurement_noise_scale_r=1.0,
                     max_frames_for_aggressive_reid=5, aggressive_reid_distance_factor=5.0):
        """
        Initialize the offline foosball tracker.

        Args:
            conf_threshold (float): Minimum confidence for a detection to be considered.
            max_lost_frames (int): Maximum frames a track can be 'lost' before termination.
                                    Corresponds to ~1 second at 30 FPS.
            min_init_detections (int): Number of consecutive high-confidence detections
                                        required to initiate a new track.
            reid_proximity_thresh (float): Maximum pixel distance for standard re-identifying a lost track.
            process_noise_scale_q (float): Scale for Kalman Filter's process noise (Q).
            measurement_noise_scale_r (float): Scale for Kalman Filter's measurement noise (R).
            max_frames_for_aggressive_reid (int): Max frames a track can be lost for aggressive re-identification.
            aggressive_reid_distance_factor (float): Multiplier for reid_proximity_thresh for aggressive re-ID.
        """
        self.conf_threshold = conf_threshold
        self.max_lost_frames = max_lost_frames
        self.min_init_detections = min_init_detections
        self.reid_proximity_thresh = reid_proximity_thresh # pixels
        
        # Store Kalman filter parameters to pass to individual tracks
        self.process_noise_scale_q = process_noise_scale_q
        self.measurement_noise_scale_r = measurement_noise_scale_r
        
        # New parameters for aggressive re-identification
        self.max_frames_for_aggressive_reid = max_frames_for_aggressive_reid
        # Calculate the aggressive re-ID threshold
        self.aggressive_reid_distance_threshold = reid_proximity_thresh * aggressive_reid_distance_factor

        self.active_tracks = []
        self.terminated_tracks = []
        self.potential_new_tracks = deque() # Stores recent detections to check for new tracks
        self.frame_to_tracked_detections_map = {} # Stores results per frame after processing
        self.all_tracks_by_id = {} # New: Stores all track objects indexed by their ID

    def euclidean_distance(self, p1, p2):
        return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

    def process_video(self, all_detections_by_frame):
        """
        Processes all detections for the entire video to generate robust tracks.

        Args:
            all_detections_by_frame (dict): A dictionary where keys are frame indices
                                            and values are lists of detection dicts
                                            (e.g., {'bbox': [x1,y1,x2,y2], 'confidence': 0.9}).
        Returns:
            dict: A dictionary where keys are frame indices and values are lists of
                  (Detection, track_id) tuples.
        """
        num_frames = max(all_detections_by_frame.keys()) + 1 if all_detections_by_frame else 0

        for frame_idx in tqdm(range(num_frames), desc="Processing Frames for Tracking"):
            current_frame_raw_detections = all_detections_by_frame.get(frame_idx, [])
            current_frame_detections = [
                Detection(frame_idx, d['bbox'], d['confidence'])
                for d in current_frame_raw_detections if d['confidence'] >= self.conf_threshold
            ]

            # Store the tracked detections for the current frame
            self.frame_to_tracked_detections_map[frame_idx] = []

            # Initialize sets to keep track of matched detections for the current frame
            matched_detections_indices = set()
            
            # --- Phase 1: Match current detections to ACTIVE tracks (using Kalman predicted position) ---
            # Prioritize matching to currently active tracks
            for track in self.active_tracks:
                if track.state == 'active':
                    predicted_pos = track.get_predicted_position()
                    min_dist = float('inf')
                    best_match_idx = -1

                    for i, det in enumerate(current_frame_detections):
                        if i in matched_detections_indices:
                            continue

                        dist = self.euclidean_distance(predicted_pos, det.center)
                        # Use a standard re-identification threshold for active tracks
                        if dist < min_dist and dist < self.reid_proximity_thresh: 
                            min_dist = dist
                            best_match_idx = i

                    if best_match_idx != -1:
                        track.add_detection(current_frame_detections[best_match_idx])
                        self.frame_to_tracked_detections_map[frame_idx].append((current_frame_detections[best_match_idx], track.track_id))
                        matched_detections_indices.add(best_match_idx)

            # Collect unmatched detections after Phase 1
            unmatched_detections = [
                det for i, det in enumerate(current_frame_detections)
                if i not in matched_detections_indices
            ]
            
            # --- Phase 2: Aggressive Re-identification for RECENTLY LOST tracks ---
            # This phase handles tracks that *just* became lost or were lost for a short period
            # and might reappear in a distant location due to a quick shot, or just a momentary occlusion.
            # This is where we prevent new tracks from being created if an old one can be re-identified.
            
            matched_unmatched_indices_aggressive = set() # To track which of the unmatched_detections are used in this phase

            # Sort unmatched detections by confidence (descending) to prioritize high-conf ones for re-ID
            unmatched_detections.sort(key=lambda x: x.confidence, reverse=True)

            for track in self.active_tracks: # Active tracks might include 'lost' ones from previous frame
                if track.state == 'lost' and track.frames_since_last_detection <= self.max_frames_for_aggressive_reid:
                    min_dist_aggressive_reid = float('inf')
                    best_unmatched_det_idx = -1

                    for j, unmatched_det in enumerate(unmatched_detections):
                        if j in matched_unmatched_indices_aggressive: # Detection already matched in this phase
                            continue

                        # Check distance against last known position for aggressive re-ID
                        dist = self.euclidean_distance(track.last_known_position, unmatched_det.center)
                        if dist < min_dist_aggressive_reid and dist < self.aggressive_reid_distance_threshold:
                            min_dist_aggressive_reid = dist
                            best_unmatched_det_idx = j
                    
                    if best_unmatched_det_idx != -1:
                        track.add_detection(unmatched_detections[best_unmatched_det_idx])
                        self.frame_to_tracked_detections_map[frame_idx].append((unmatched_detections[best_unmatched_det_idx], track.track_id))
                        matched_unmatched_indices_aggressive.add(best_unmatched_det_idx)
            
            # Collect truly unmatched detections after Phase 2
            unmatched_detections_after_reid = [
                det for j, det in enumerate(unmatched_detections)
                if j not in matched_unmatched_indices_aggressive
            ]

            # --- Phase 3: Update track states (mark lost, terminate) ---
            tracks_to_remove = []
            for track in self.active_tracks:
                if track.last_detection_frame < frame_idx: # If no detection added in this frame
                    track.mark_lost()
                    if track.frames_since_last_detection > self.max_lost_frames:
                        track.state = 'terminated'
                        self.terminated_tracks.append(track)
                        tracks_to_remove.append(track)
            
            self.active_tracks = [t for t in self.active_tracks if t.state != 'terminated']

            # --- Phase 4: Initiate new tracks from remaining unmatched detections ---
            for det in unmatched_detections_after_reid: # Use the detections that truly remain unmatched
                self.potential_new_tracks.append(det)

            # Filter potential_new_tracks to only include detections from recent frames
            while self.potential_new_tracks and \
                    self.potential_new_tracks[0].frame_idx < frame_idx - (self.min_init_detections + self.max_frames_for_aggressive_reid + 5):
                # Added max_frames_for_aggressive_reid buffer to ensure lost tracks have a chance to re-identify
                self.potential_new_tracks.popleft()

            newly_started_tracks_indices = []
            for i in range(len(self.potential_new_tracks)):
                current_det = self.potential_new_tracks[i]
                consecutive_count = 1
                temp_detections_for_init = [current_det]
                
                for j in range(i + 1, len(self.potential_new_tracks)):
                    next_det = self.potential_new_tracks[j]
                    if next_det.frame_idx == current_det.frame_idx + consecutive_count and \
                       self.euclidean_distance(current_det.center, next_det.center) < 150: # Max pixel jump for consecutive
                        consecutive_count += 1
                        temp_detections_for_init.append(next_det)
                    else:
                        break

                if consecutive_count >= self.min_init_detections:
                    # Pass Kalman filter parameters to the new Track instance
                    new_track = Track(temp_detections_for_init[0], 
                                      self.process_noise_scale_q, self.measurement_noise_scale_r)
                    self.active_tracks.append(new_track)
                    self.all_tracks_by_id[new_track.track_id] = new_track # Store new track for lookup

                    for k in range(1, len(temp_detections_for_init)):
                        new_track.add_detection(temp_detections_for_init[k])
                    
                    # Add initial detections of new track to current frame's output
                    for init_det in temp_detections_for_init:
                        if init_det.frame_idx == frame_idx:
                            self.frame_to_tracked_detections_map[frame_idx].append((init_det, new_track.track_id))
                    
                    newly_started_tracks_indices.append(i)

            for idx in sorted(newly_started_tracks_indices, reverse=True):
                del self.potential_new_tracks[idx]
        
        return self.frame_to_tracked_detections_map

def save_tracked_data_to_json(tracked_results_by_frame, output_json_path, frame_width, frame_height):
    """
    Saves the tracked detection data to a JSON file.

    Args:
        tracked_results_by_frame (dict): Dictionary with tracked detections per frame.
        output_json_path (str): Path to save the JSON file.
        frame_width (int): Width of the video frames.
        frame_height (int): Height of the video frames.
    """
    json_output_data = []
    
    for frame_idx in sorted(tracked_results_by_frame.keys()):
        frame_data = {
            "frame_count": frame_idx,
            "boxes": [],
            "current_ball_location": None, # Default to None, update if a tracked ball is found
            "ball_status": "Unknown" # Placeholder, as this requires higher-level logic
        }
        
        best_ball_confidence = -1
        best_ball_bbox = None

        for det, track_id in tracked_results_by_frame[frame_idx]:
            # Convert pixel bbox back to normalized [x_center, y_center, width, height]
            x_min, y_min, x_max, y_max = det.bbox
            
            width = (x_max - x_min) / frame_width
            height = (y_max - y_min) / frame_height
            x_center = (x_min + x_max) / 2 / frame_width
            y_center = (y_min + y_max) / 2 / frame_height

            box_entry = {
                "bbox": [x_center, y_center, width, height],
                "conf": det.confidence,
                "cls": 0, # Assuming class 0 is always ball
                "id": track_id
            }
            frame_data["boxes"].append(box_entry)

            # Update current_ball_location with the highest confidence tracked ball
            if det.confidence > best_ball_confidence:
                best_ball_confidence = det.confidence
                best_ball_bbox = box_entry["bbox"] # Use normalized bbox for current_ball_location

        if best_ball_bbox:
            frame_data["current_ball_location"] = best_ball_bbox

        json_output_data.append(frame_data)
    
    print(f"Saving tracked data to {output_json_path}...")
    with open(output_json_path, 'w') as f:
        json.dump(json_output_data, f, indent=4)
    print("Tracked data saved successfully.")


# --- Main Visualization Script ---

def visualize_foosball_tracking(video_path, detection_json_path, output_video_path, tracker_params, trail_length_frames=60, output_tracked_json_path=None):
    # Load raw detections from JSON
    print("Loading detections from JSON...")
    with open(detection_json_path, 'r') as f:
        raw_detections_data = json.load(f)

    # Prepare detections for the tracker, converting normalized bbox to pixel coordinates
    # and structuring by frame_idx
    all_detections_by_frame = {}
    
    # Get video properties to convert normalized bounding boxes
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video file: {video_path}")
    
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"Video Info: {frame_width}x{frame_height} @ {fps:.2f} FPS, {total_frames} frames")

    # Assuming the JSON provides detections as a list of objects, one per frame
    # and each object contains 'boxes' list
    for frame_data in raw_detections_data:
        frame_idx = frame_data['frame_count']
        current_frame_detections = []
        for box_data in frame_data['boxes']:
            if box_data['cls'] == 0: # Only interested in 'ball' detections (class 0)
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
                    'confidence': box_data['conf']
                })
        all_detections_by_frame[frame_idx] = current_frame_detections
    print(f"Loaded detections for {len(all_detections_by_frame)} frames.")


    # Initialize and run the OfflineFoosballTracker with passed parameters
    tracker = OfflineFoosballTracker(**tracker_params)
    print("Running offline tracking...")
    tracked_results_by_frame = tracker.process_video(all_detections_by_frame)
    print("Tracking complete.")

    # Save tracked data to JSON if path is provided
    if output_tracked_json_path:
        save_tracked_data_to_json(tracked_results_by_frame, output_tracked_json_path, frame_width, frame_height)

    # Prepare video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') # Codec for .mp4 files
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))
    if not out.isOpened():
        raise IOError(f"Could not open video writer for: {output_video_path}")

    print(f"Writing annotated video to: {output_video_path}")
    
    # Define a set of distinct colors for tracks (BGR format)
    # These colors are chosen to be visibly different
    track_color_palette = [
        (0, 255, 0),    # Green
        (255, 0, 0),    # Blue
        (0, 0, 255),    # Red
        (0, 255, 255),  # Yellow
        (255, 255, 0),  # Cyan
        (255, 0, 255),  # Magenta
        (128, 0, 128),  # Purple
        (0, 128, 128),  # Teal
        (128, 128, 0),  # Olive
        (0, 165, 255)   # Orange
    ]
    track_colors = {} # Maps track_id to a color

    current_frame_idx = 0
    with tqdm(total=total_frames, desc="Annotating Video") as pbar:
        while True:
            ret, frame = cap.read()
            if not ret:
                break # End of video

            # Get tracked detections for the current frame
            tracked_dets_for_frame = tracked_results_by_frame.get(current_frame_idx, [])

            # --- Draw Track Trails ---
            # Get all tracks (active or lost) to draw their history
            for track_id, track_obj in tracker.all_tracks_by_id.items():
                # Drawing all tracks regardless of 'terminated' state for full visualization
                if track_obj.detections:
                    trail_points_with_frames = []
                    # Collect points for the trail within the specified length, in chronological order
                    for det in track_obj.detections:
                        # Only include detections that have occurred up to the current frame
                        # and are within the 'trail_length_frames' window.
                        if det.frame_idx <= current_frame_idx:
                            if (current_frame_idx - det.frame_idx) < trail_length_frames:
                                trail_points_with_frames.append((det.center, det.frame_idx))
                        
                    # Ensure chronological order. This is crucial for drawing connected lines correctly.
                    # The detections in track_obj.detections are already chronological,
                    # but sorting filtered sub-lists is a good safeguard if out-of-order adds occurred.
                    trail_points_with_frames.sort(key=lambda x: x[1]) # Sort by frame_idx


                    if len(trail_points_with_frames) > 1:
                        # Assign color for the track if not already assigned
                        if track_id not in track_colors:
                            track_colors[track_id] = track_color_palette[track_id % len(track_color_palette)]
                        
                        base_color = track_colors[track_id] # (B, G, R)
                        
                        # Draw connected line segments for the trail with gradient color
                        for i in range(1, len(trail_points_with_frames)):
                            p1_center, p1_frame_idx = trail_points_with_frames[i-1]
                            p2_center, p2_frame_idx = trail_points_with_frames[i]

                            # Calculate age for this segment (using the end point of the segment for fading)
                            segment_age = current_frame_idx - p2_frame_idx
                            
                            # Normalize age (0.0 = newest, 1.0 = oldest within trail_length_frames)
                            age_ratio = float(segment_age) / max(1, trail_length_frames - 1)
                            
                            # Invert for fading: 1.0 = full color (newest), 0.0 = black (oldest)
                            fade_factor = max(0.4, 1.0 - age_ratio) # Ensure a minimum visibility of 40%

                            # Interpolate color towards black (or a darker shade)
                            faded_color = (
                                int(base_color[0] * fade_factor),
                                int(base_color[1] * fade_factor),
                                int(base_color[2] * fade_factor)
                            )
                            
                            cv2.line(frame, (int(p1_center[0]), int(p1_center[1])),
                                     (int(p2_center[0]), int(p2_center[1])),
                                     faded_color, 4) # Thickness 4 for visibility

            # --- Draw Current Frame Detections (Bounding Box, ID, Confidence) ---
            for det, track_id in tracked_dets_for_frame:
                x1, y1, x2, y2 = det.bbox
                conf = det.confidence

                # Get track color
                if track_id not in track_colors:
                    track_colors[track_id] = track_color_palette[track_id % len(track_color_palette)]
                current_track_color = track_colors[track_id]

                # Draw bounding box
                thickness = 2
                cv2.rectangle(frame, (x1, y1), (x2, y2), current_track_color, thickness)

                # Draw text: ID | Conf
                label = f"ID: {track_id} | Conf: {conf:.2f}"
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.7
                font_thickness = 2
                text_size = cv2.getTextSize(label, font, font_scale, font_thickness)[0]
                
                # Position text above the bounding box
                text_x = x1
                text_y = y1 - 10 if y1 - 10 > 10 else y1 + text_size[1] + 10 # Adjust if too high

                # Draw text background for better readability
                cv2.rectangle(frame, (text_x, text_y - text_size[1] - 5), 
                              (text_x + text_size[0] + 5, text_y + 5), (0, 0, 0), -1) # Black background
                cv2.putText(frame, label, (text_x, text_y), font, font_scale, (255, 255, 255), font_thickness, cv2.LINE_AA) # White text

            out.write(frame)
            current_frame_idx += 1
            pbar.update(1)

    cap.release()
    out.release()
    print("Video annotation complete.")
    print(f"Output video saved to: {output_video_path}")


if __name__ == "__main__":
    # --- Configuration ---
    # IMPORTANT: Replace these paths with your actual video and JSON file paths
    
    input_video_path = "/home/freystec/foosball-statistics/foosball-videos/SideKick Harburg - SiMo Doppel  05.08.2025_.mp4"
    detection_json_path = "/home/freystec/SiMo_yolov11n_imgsz_640_long.json"
    output_video_path = "/home/freystec/SiMo_yolov11n_imgsz_640_long_Tracked.mp4"
    output_tracked_json_path = "/home/freystec/SiMo_yolov11n_imgsz_640_long_tracked_data.json" # Path for JSON output
    # input_video_path = "/home/freystec/foosball-statistics/foosball-videos/topview_leo_championship_match2.mp4"
    # detection_json_path = "/home/freystec/topview_leo_championship_match2_detections.json"
    # output_video_path = "/home/freystec/topview_leo_championship_match2_tracked.mp4"
    # output_tracked_json_path = "/home/freystec/topview_leo_championship_match2_tracked_data.json" # New path for JSON output
    # Tracker Initialization Parameters
    tracker_parameters = {
        "conf_threshold": 0.7,
        "max_lost_frames": 60, # Allows track to be lost for ~2 seconds at 30 FPS
        "min_init_detections": 6, # Requires 6 consecutive high-confidence detections to start a track
        "reid_proximity_thresh": 1000, # Standard re-ID distance for re-identifying a lost track (pixels)
        "process_noise_scale_q": 100.0, # Kalman Filter Q scale: higher for fast changes
        "measurement_noise_scale_r": 0.1, # Kalman Filter R scale: lower for reliable detections
        "max_frames_for_aggressive_reid": 30, # Max frames a track can be lost for aggressive re-ID (e.g., 0.5s at 30 FPS)
        "aggressive_reid_distance_factor": 5.0 # Multiplier for reid_proximity_thresh for aggressive re-ID (e.g., 5x standard)
    }

    # Trail length in frames (e.g., 60 frames = 2 seconds at 30 FPS)
    desired_trail_length_frames = 60

    try:
        visualize_foosball_tracking(input_video_path, detection_json_path, output_video_path,
                                     tracker_parameters, desired_trail_length_frames, output_tracked_json_path)
    except Exception as e:
        print(f"An error occurred: {e}")
        print("Please ensure your video path and detection JSON path are correct and accessible.")
        print("Also, make sure 'opencv-python' and 'tqdm' are installed (`pip install opencv-python tqdm`).")