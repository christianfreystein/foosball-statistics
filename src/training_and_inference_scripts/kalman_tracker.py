from filterpy.kalman import KalmanFilter
import numpy as np
import supervision as sv
from collections import deque
from ultralytics import YOLO
import cv2
from tqdm import tqdm
import time

class KalmanBallTracker:
    # Define states as constants for clarity
    NOT_TRACKING = 0
    INITIALIZING = 1
    TRACKING = 2
    LOST = 3

    def __init__(self, dt: float = 1.0, 
                 process_noise_scale: float = 5.0, # Increased for fast movements
                 measurement_noise_scale: float = 2.0, # Lower, as detections are reliable
                 conf_threshold: float = 0.7, # Higher confidence for filtering detections
                 gating_threshold: float = 70.0, # Max pixels from prediction for association
                 min_detections_to_init: int = 3, # Detections needed to start a track
                 max_frames_to_maintain: int = 15, # Frames to predict without detection before going to LOST
                 re_id_distance_threshold: float = 100.0, # Max pixels for re-identification after loss
                 re_id_max_frames: int = 60, # Max frames to try re-identifying (e.g., 2 seconds at 30 FPS)
                 output_bbox_size: int = 20 # Fixed size for the output bounding box
                 ):
        
        self.kf = None
        self.dt = dt
        self.process_noise_scale = process_noise_scale
        self.measurement_noise_scale = measurement_noise_scale
        self.conf_threshold = conf_threshold
        self.gating_threshold = gating_threshold
        
        self.track_state = self.NOT_TRACKING
        self.frames_since_last_detection = 0 # In TRACKING state
        self.lost_frame_count = 0 # In LOST state

        self.min_detections_to_init = min_detections_to_init
        self.init_detections_buffer = deque(maxlen=min_detections_to_init)
        
        self.max_frames_to_maintain = max_frames_to_maintain
        self.re_id_distance_threshold = re_id_distance_threshold
        self.re_id_max_frames = re_id_max_frames
        
        self._last_known_state = None # To store KF.x when moving to LOST state
        self.output_bbox_size = output_bbox_size # Fixed size for drawing the output bbox

    def _initialize_kalman_filter(self, initial_position: np.ndarray, initial_velocity: np.ndarray):
        """Initializes the Kalman filter with a given state."""
        self.kf = KalmanFilter(dim_x=4, dim_z=2)
        self.kf.x = np.array([initial_position[0], initial_position[1], initial_velocity[0], initial_velocity[1]]).reshape(4, 1)
        self.kf.F = np.array([[1, 0, self.dt, 0], [0, 1, 0, self.dt], [0, 0, 1, 0], [0, 0, 0, 1]])
        self.kf.H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]])
        
        self.kf.Q = np.eye(4) * self.process_noise_scale
        self.kf.Q[0,0] = self.process_noise_scale * (self.dt**3)/3
        self.kf.Q[1,1] = self.process_noise_scale * (self.dt**3)/3
        self.kf.Q[2,2] = self.process_noise_scale * self.dt
        self.kf.Q[3,3] = self.process_noise_scale * self.dt

        self.kf.R = np.eye(2) * self.measurement_noise_scale
        self.kf.P = np.eye(4) * 1000. # High initial uncertainty

    def _get_tracked_detection(self, confidence: float = 1.0) -> sv.Detections:
        """Helper to create a sv.Detections object from the current KF state."""
        # Ensure KF exists AND is in a state where it has a valid current prediction/state
        # This prevents drawing from a 'None' KF or when it's not truly tracking
        if self.kf is None or (self.track_state != self.TRACKING and self.track_state != self.LOST):
            return sv.Detections.empty()

        # FIX: Ensure current_x and current_y are scalars (as discussed in previous steps)
        current_x = self.kf.x[0, 0]
        current_y = self.kf.x[1, 0]

        half_size = self.output_bbox_size / 2
        x1 = current_x - half_size
        y1 = current_y - half_size
        x2 = current_x + half_size
        y2 = current_y + half_size

        tracked_bbox = np.array([[x1, y1, x2, y2]])
        tracked_confidence = np.array([confidence])
        tracked_class_id = np.array([0])
        return sv.Detections(xyxy=tracked_bbox, confidence=tracked_confidence, class_id=tracked_class_id)

    def update(self, detections: sv.Detections) -> sv.Detections:
        # 1. Filter detections by confidence
        filtered_detections = detections[detections.confidence > self.conf_threshold]
        
        # Get candidate centers from filtered_detections
        candidate_centers = []
        for det_box in filtered_detections.xyxy:
            candidate_centers.append([(det_box[0] + det_box[2]) / 2, (det_box[1] + det_box[3]) / 2])
        candidate_centers = np.array(candidate_centers)

        # Initialize variables
        predicted_pos = None 
        associated_detection_idx = -1 # Index of the best detection for current update
        
        # --- State Machine Logic ---
        
        # State: TRACKING
        if self.track_state == self.TRACKING:
            self.kf.predict() # KF still predicts internally for state update and next prediction
            predicted_pos = self.kf.x[0:2].flatten() # Get predicted position after prediction

            self.frames_since_last_detection += 1 
            
            # Try to associate with prediction
            if predicted_pos is not None and len(candidate_centers) > 0:
                distances = np.linalg.norm(candidate_centers - predicted_pos, axis=1)
                min_distance_idx = np.argmin(distances)
                
                if distances[min_distance_idx] < self.gating_threshold:
                    associated_detection_idx = min_distance_idx
            
            if associated_detection_idx != -1:
                # Associated successfully -> Update KF
                best_det_center = candidate_centers[associated_detection_idx].reshape(2, 1)
                self.kf.update(best_det_center)
                self.frames_since_last_detection = 0
                # Return the detection based on the *updated* KF state (which used a true detection)
                return self._get_tracked_detection(confidence=1.0) 

            else: # No detection associated in this frame
                if self.frames_since_last_detection > self.max_frames_to_maintain:
                    # Transition to LOST state
                    print("Track transitioning to LOST state.")
                    self.track_state = self.LOST
                    self._last_known_state = self.kf.x.copy() # Store last known state *before* losing track
                    self.lost_frame_count = 0
                    self.kf = None # Nullify KF when LOST so it doesn't predict further
                    return sv.Detections.empty() # No output when transitioning to LOST
                
                else: 
                    # CRITICAL CHANGE HERE:
                    # Ball is unseen, but still within maintenance window.
                    # We continue to predict internally, but DO NOT return this predicted position for visualization.
                    return sv.Detections.empty() # Return empty detections, so nothing is drawn
                                                 # This stops printing "guessed annotations".

        # State: LOST
        elif self.track_state == self.LOST:
            self.lost_frame_count += 1
            # No kf.predict() here. Rely on _last_known_state for re-ID.

            # Try to re-identify
            if len(candidate_centers) > 0 and self._last_known_state is not None:
                re_id_pos = self._last_known_state[0:2].flatten() # This is the *remembered* position
                distances = np.linalg.norm(candidate_centers - re_id_pos, axis=1)
                min_distance_idx = np.argmin(distances)

                if distances[min_distance_idx] < self.re_id_distance_threshold:
                    # Re-identified successfully!
                    print(f"Track re-identified after {self.lost_frame_count} frames.")
                    re_id_det_center = candidate_centers[min_distance_idx]

                    initial_velocity = self._last_known_state[2:4].flatten()
                    self._initialize_kalman_filter(re_id_det_center, initial_velocity)
                    self.kf.update(re_id_det_center.reshape(2,1)) # Update with current detection
                    self.track_state = self.TRACKING
                    self.frames_since_last_detection = 0
                    self.lost_frame_count = 0
                    self._last_known_state = None
                    # Return the detection based on the *re-initialized and updated* KF state
                    return self._get_tracked_detection(confidence=1.0)

            if self.lost_frame_count > self.re_id_max_frames:
                print("Track completely lost (re-identification window expired).")
                self.track_state = self.NOT_TRACKING
                self.init_detections_buffer.clear()
                self._last_known_state = None

            return sv.Detections.empty() # No active track to return when LOST and not re-identified

        # State: INITIALIZING
        elif self.track_state == self.INITIALIZING:
            if len(filtered_detections) == 0: # If no detection in this frame, reset initialization
                print("Initialization reset due to no detections.")
                self.init_detections_buffer.clear()
                self.track_state = self.NOT_TRACKING
                return sv.Detections.empty()
            
            # Take the highest confidence detection for initialization
            best_init_det_idx = np.argmax(filtered_detections.confidence)
            best_init_det_center = candidate_centers[best_init_det_idx]
            self.init_detections_buffer.append(best_init_det_center)

            if len(self.init_detections_buffer) == self.min_detections_to_init:
                # Enough detections, initialize Kalman Filter
                initial_positions = list(self.init_detections_buffer) # Copy buffer contents
                
                # Calculate average position
                initial_position = np.mean(initial_positions, axis=0)
                
                # Estimate initial velocity based on the first and last detection in buffer
                if self.min_detections_to_init > 1:
                    initial_velocity = (initial_positions[-1] - initial_positions[0]) / (self.dt * (self.min_detections_to_init - 1))
                else:
                    initial_velocity = np.array([0., 0.])
                
                self._initialize_kalman_filter(initial_position, initial_velocity)
                self.track_state = self.TRACKING
                self.frames_since_last_detection = 0
                self.init_detections_buffer.clear() # Clear buffer after successful initialization
                print("Track initialized successfully.")
                
                # Now that KF is initialized, return the first prediction/update
                # (You might want to immediately update with the current detection if it's the one that triggered init)
                # For simplicity here, we'll let the next frame's update cycle handle it.
                return self._get_tracked_detection(confidence=1.0) 
            
            return sv.Detections.empty() # Still initializing

        # State: NOT_TRACKING (Default)
        elif self.track_state == self.NOT_TRACKING:
            if len(filtered_detections) > 0:
                # Start initialization if any high-confidence detection appears
                print("Starting initialization phase.")
                self.track_state = self.INITIALIZING
                # Add the first candidate detection to the buffer for initialization
                # (Take the highest confidence one if multiple)
                best_init_det_idx = np.argmax(filtered_detections.confidence)
                best_init_det_center = candidate_centers[best_init_det_idx]
                self.init_detections_buffer.append(best_init_det_center)
            
            return sv.Detections.empty() # No track to return yet
        
        return sv.Detections.empty() # Fallback, should not be reached normally



class BallAnnotator:
    def __init__(
            self,
            radius: int,
            buffer_size: int = 5,
            thickness: int = 2
    ):
        self.color_palette = sv.ColorPalette.from_matplotlib('jet', buffer_size)
        self.buffer = deque(maxlen=buffer_size)
        self.radius = radius
        self.thickness = thickness

    def interpolate_radius(self, i: int, max_i: int) -> int:
        if max_i == 1:
            return self.radius
        return int(1 + i * (self.radius - 1) / (max_i - 1))

    def annotate(
            self,
            frame: np.ndarray,
            detections: sv.Detections
    ) -> np.ndarray:
        xy = detections.get_anchors_coordinates(
            sv.Position.BOTTOM_CENTER).astype(int)
        self.buffer.append(xy)

        for i, xy in enumerate(self.buffer):
            color = self.color_palette.by_idx(i)
            interpolated_radius = self.interpolate_radius(i, len(self.buffer))

            for center in xy:
                frame = cv2.circle(
                    img=frame,
                    center=tuple(center),
                    radius=interpolated_radius,
                    color=color.as_bgr(),
                    thickness=self.thickness
                )
        return frame


model = YOLO(r"/home/freystec/repositories/yolo-distiller/runs/detect/train/weights/yolov11n_with_KD_cropped.pt")
source_path = (r"/home/freystec/foosball-statistics/foosball-videos/Leonhart_clip.mp4")
target_path = (r"/home/freystec/speed_test.mp4")

video_info = sv.VideoInfo.from_video_path(source_path)
frame_generator = sv.get_video_frames_generator(source_path)
w, h = video_info.width, video_info.height

# --- Define Crop Parameters ---
# Adjust these values based on your video to focus on the foosball table
crop_left = 150  # pixels to cut from the left
crop_right = 150 # pixels to cut from the right
crop_top = 0    # pixels to cut from the top
crop_bottom = 0 # pixels to cut from the bottom

# Calculate the dimensions of the cropped region for validation
cropped_w = w - crop_left - crop_right
cropped_h = h - crop_top - crop_bottom

# Ensure crop dimensions are valid to avoid errors
if cropped_w <= 0 or cropped_h <= 0:
    raise ValueError(f"Invalid crop dimensions. Cropped width: {cropped_w}, Cropped height: {cropped_h}. "
                     "Please check your crop_left, crop_right, crop_top, crop_bottom values. "
                     "Cropped dimensions must be positive.")

annotator = BallAnnotator(20,30,4) # Assuming these parameters are appropriate for the original frame size

tracker = KalmanBallTracker(
    dt=1.0,
    process_noise_scale=100.0, # Experiment with this, high for fast changes
    measurement_noise_scale=2.0, # Relatively low, as detections are good
    conf_threshold=0.6, # Consider only YOLO detections with >70% confidence
    gating_threshold=600.0, # Max pixels from prediction for association
    min_detections_to_init=6, # Need 3 consecutive high-conf detections to start
    max_frames_to_maintain=30, # Maintain track for 15 frames (~0.5 sec at 30 FPS)
    re_id_distance_threshold=150.0, # Allow re-id up to 150 pixels from last known spot
    re_id_max_frames=90, # Try re-identifying for up to 60 frames (2 seconds)
    output_bbox_size=25 # Example fixed size for output bounding box
)

# Initialize accumulators for time for each step
total_time_step1_crop = 0.0
total_time_step2_detection = 0.0
total_time_step3_sv_conversion = 0.0
total_time_step4_adjustment = 0.0
total_time_step5_tracking = 0.0
total_time_step6_annotation = 0.0
total_time_step7_sink_write = 0.0


with sv.VideoSink(target_path, video_info=video_info) as sink:
    for frame_idx, frame in enumerate(tqdm(frame_generator, total=video_info.total_frames, desc="Processing Video")):
        # Step 1: Crop the frame for detection
        start_time = time.perf_counter()
        cropped_frame = frame[crop_top : h - crop_bottom, crop_left : w - crop_right]
        end_time = time.perf_counter()
        total_time_step1_crop += (end_time - start_time)

        # Step 2: Perform detection on the cropped frame
        start_time = time.perf_counter()
        results = model(cropped_frame, verbose=False)[0]
        end_time = time.perf_counter()
        total_time_step2_detection += (end_time - start_time)

        # Step 3: Convert results to sv.Detections
        start_time = time.perf_counter()
        detections = sv.Detections.from_ultralytics(results)
        end_time = time.perf_counter()
        total_time_step3_sv_conversion += (end_time - start_time)
        
        # Step 4: MANUAL ADJUSTMENT: Adjust bounding boxes back to original frame coordinates
        start_time = time.perf_counter()
        if len(detections.xyxy) > 0:
            detections.xyxy[:, 0] += crop_left
            detections.xyxy[:, 2] += crop_left
            detections.xyxy[:, 1] += crop_top
            detections.xyxy[:, 3] += crop_top
        end_time = time.perf_counter()
        total_time_step4_adjustment += (end_time - start_time)

        # Step 5: Update the tracker with the detections in original coordinates
        start_time = time.perf_counter()
        tracked_detections = tracker.update(detections)
        end_time = time.perf_counter()
        total_time_step5_tracking += (end_time - start_time)
        
        # Step 6: Annotate the ORIGINAL frame with the correctly positioned (tracked) detections
        start_time = time.perf_counter()
        frame = annotator.annotate(frame, tracked_detections)
        end_time = time.perf_counter()
        total_time_step6_annotation += (end_time - start_time)
        
        # Step 7: Write the annotated original frame to the video sink
        start_time = time.perf_counter()
        sink.write_frame(frame)
        end_time = time.perf_counter()
        total_time_step7_sink_write += (end_time - start_time)

print(f"Video processed and saved to: {target_path}\n")

# Calculate and print average times
num_frames = video_info.total_frames
print("--- Average Time per Frame (in seconds) ---")
print(f"1. Crop Frame:           {total_time_step1_crop / num_frames:.5f} s")
print(f"2. YOLO Detection:       {total_time_step2_detection / num_frames:.5f} s")
print(f"3. SV Detections Conv:   {total_time_step3_sv_conversion / num_frames:.5f} s")
print(f"4. BBox Adjustment:      {total_time_step4_adjustment / num_frames:.5f} s")
print(f"5. Kalman Tracking:      {total_time_step5_tracking / num_frames:.5f} s")
print(f"6. Annotation:           {total_time_step6_annotation / num_frames:.5f} s")
print(f"7. Video Sink Write:     {total_time_step7_sink_write / num_frames:.5f} s")

# You can also print total processing time for all steps combined
total_processing_time = (total_time_step1_crop + total_time_step2_detection +
                         total_time_step3_sv_conversion + total_time_step4_adjustment +
                         total_time_step5_tracking + total_time_step6_annotation +
                         total_time_step7_sink_write)
print(f"\nTotal Processing Time (excluding file I/O setup): {total_processing_time:.2f} seconds")
print(f"Average FPS (based on measured steps): {num_frames / total_processing_time:.2f} FPS")

