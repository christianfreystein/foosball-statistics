import time
import json
import os
import cv2
from datetime import datetime
from Object_Detector import Detector, process_stream  # Assuming your Detector class is in a file named detector.py


if __name__ == "__main__":
    # Define paths and parameters
    model_path = r"C:\Users\chris\foosball-statistics\weights\yolov11n_imgsz640_Topview.pt"
    stream_url = "rtmp://localhost/live/teststream"
    json_path = "output_predictions.json"
    output_annotated_video = "output_video.mp4"

    # Define crop parameters
    crop_params = {'left': 300, 'right': 300, 'top': 0, 'bottom': 0}

    # Initialize the detector
    try:
        detector = Detector(model_path, crop_params)
    except Exception as e:
        print(f"Error initializing detector: {e}")
        exit()

    print(f"Starting to process stream from: {stream_url}")
    start_time = time.time()

    # Process the video stream
    full_data, processed_frames = process_stream(
        stream_source=stream_url,
        detector=detector,
        save_annotated_video=True,
        output_video_path=output_annotated_video
    )

    end_time = time.time()

    if full_data and processed_frames:
        print(f"Stream processing completed in {(end_time - start_time):.2f} seconds.")
        print(f"The total number of frames processed is: {len(processed_frames)}")

        # Save predictions to a JSON file
        try:
            with open(json_path, 'w') as f:
                json.dump(full_data, f, indent=4)
            print(f"Predictions saved to {json_path}")
        except Exception as e:
            print(f"Error saving JSON file: {e}")
    else:
        print("Processing failed or no frames were processed.")