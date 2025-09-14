import time
import json
import cv2
from Object_Detector import Detector, process_video
from Offline_Tracker import run_tracking_on_detections, save_tracked_data_to_json, visualize_tracking_on_video


def main():
    """
    Main function to orchestrate the video processing, tracking, and visualization workflow.
    """
    # Define input and output file paths
    # Note: These paths are specific to the user's local machine and should be adjusted
    # to match the user's file system if running on a different computer.
    model_path = r"C:\Users\chris\foosball-statistics\weights\yolov11n_imgsz640_Topview.pt"
    video_path = r"C:\Users\chris\foosball-statistics\foosball-videos\SideKick Harburg - SiMo Doppel  05.08.2025.mp4"
    json_path = "output_predictions.json"
    annotated_video_path = "output_video_annotated.mp4"
    output_tracked_json_path = "tracked_results.json"
    output_tracked_video_path = "output_video_tracked.mp4"

    # Define crop parameters for the detector
    crop_params = {'left': 300, 'right': 300, 'top': 0, 'bottom': 0}

    print("Step 1: Running object detection on the video...")
    # Initialize the detector with the specified model and crop parameters
    detector = Detector(model_path, crop_params)

    start_time = time.time()

    # Process the video to get object detections and save them to a JSON file
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

    # Get video dimensions and FPS from the loaded frames
    if not loaded_frames:
        print("Error: No frames were loaded from the video. Exiting.")
        return
    frame_height, frame_width, _ = loaded_frames[0].shape

    # Get the FPS from the video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file at {video_path}")
        return
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()

    # The detections_data is the same as predictions_list, so we can use it directly
    detections_data = predictions_list

    # Tracker parameters (as provided in the prompt)
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

    print("Step 2: Running tracking logic on the detection data...")
    # Run the tracking logic on the loaded detection data
    tracked_results, tracker_obj = run_tracking_on_detections(detections_data, frame_width, frame_height,
                                                              tracker_parameters)

    # Save the tracked data to a new JSON file
    save_tracked_data_to_json(tracked_results, output_tracked_json_path, frame_width, frame_height)
    print(f"Tracked data saved to '{output_tracked_json_path}'.")

    print("Step 3: Visualizing tracked information on the video...")
    # Visualize the tracked information on the original video and save the result
    annotated_frames = visualize_tracking_on_video(frames=loaded_frames, fps=fps,
                                                   tracked_results_by_frame=tracked_results,
                                                   tracker=tracker_obj,
                                                   trail_length_frames=desired_trail_length_frames,
                                                   output_video_path=output_tracked_video_path)

    print(f"Completed! The final annotated video is saved at '{output_tracked_video_path}'.")
    print("The 'annotated_frames' output is now available.")


if __name__ == "__main__":
    main()
