import time
import json
import cv2
from Object_Detector import Detector, process_video
from Offline_Tracker import run_tracking_on_detections, save_data_to_json, visualize_tracking_on_video
from Calculating_Heatmap import generate_heatmap, overlay_heatmap, extract_points_from_data
import os


def main():
    """
    Main function to orchestrate the video processing, tracking, and visualization workflow.
    """
    # Define input and output file paths
    model_path = r"C:\Users\chris\foosball-statistics\weights\yolov11n_imgsz640_Topview.pt"
    video_path = r"C:\Users\chris\foosball-statistics\foosball-videos\SideKick Harburg - SiMo Doppel  05.08.2025.mp4"
    json_path = "output_predictions.json"
    annotated_video_path = "output_video_annotated.mp4"
    output_tracked_json_path = "tracked_results.json"
    output_tracked_video_path = "output_video_tracked.mp4"
    screenshot_path = r'C:\Users\chris\foosball-statistics\src\first_statistic_in_stream\first_frame.jpg'

    # Define crop parameters for the detector
    crop_params = {'left': 300, 'right': 300, 'top': 0, 'bottom': 0}

    print("Step 1: Running object detection on the video...")
    detector = Detector(model_path, crop_params)

    start_time = time.time()
    predictions_list, loaded_frames = process_video(
        video_path=video_path,
        detector=detector,
        save_json=True,
        save_annotated_video=False,
        output_video_path=annotated_video_path,
        json_path=json_path
    )
    end_time = time.time()
    print(f"Object detection completed in {end_time - start_time:.2f} seconds.")

    if not loaded_frames:
        print("Error: No frames were loaded from the video. Exiting.")
        return
    frame_height, frame_width, _ = loaded_frames[0].shape

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file at {video_path}")
        return
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()

    # Tracker parameters
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

    print("Step 2: Running tracking logic on the data...")
    merged_data, tracker_obj = run_tracking_on_detections(predictions_list, tracker_parameters)

    print("Step 3: Saving the complete, merged data...")
    if output_tracked_json_path:
        save_data_to_json(merged_data, output_tracked_json_path)
        print(f"✅ Tracked data saved successfully to '{output_tracked_json_path}'.")

    print("\nStep 4: Visualizing the tracked information on the video...")
    annotated_frames = visualize_tracking_on_video(video_path=video_path,
                                                   tracked_data=merged_data,
                                                   tracker=tracker_obj,
                                                   trail_length_frames=desired_trail_length_frames,
                                                   output_video_path=output_tracked_video_path)
    print(f"Completed! The final annotated video is saved at '{output_tracked_video_path}'.")

    # --- Part 5: Heatmap Generation from Tracked Data ---
    print("\n--- Step 5: Heatmap Generation ---")

    # Use the output from the tracking step as the input for the heatmap
    tracked_data = merged_data
    output_dir = os.path.dirname(output_tracked_json_path)

    # Extract points from the tracked data
    # The 'current_location' key is now used, as per the original heatmap code
    original_points = extract_points_from_data(tracked_data, 'current_location', frame_width, frame_height)
    print("✅ Extracted points from tracked data.")

    # Generate the heatmap
    original_heatmap_output_path = os.path.join(output_dir, 'original_ball_position_heatmap.jpg')
    original_heatmap_grid, original_heatmap_buffer = generate_heatmap(
        original_points,
        (frame_width, frame_height),
        min_count=1,
        desc_prefix="Original "
    )
    print("✅ Original heatmap generation complete.")

    # Overlay the heatmap on the screenshot
    overlaid_original_path = os.path.join(output_dir, 'overlaid_original_heatmap.jpg')
    overlay_heatmap(screenshot_path, original_heatmap_grid, overlaid_original_path, alpha=0.90)
    print(f"✅ Original heatmap overlaid on screenshot and saved to '{overlaid_original_path}'.")

    print("\n--- Workflow completed successfully! ---")


# To run this, simply call main() in your script.
if __name__ == "__main__":
    main()
