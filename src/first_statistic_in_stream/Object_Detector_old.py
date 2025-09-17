import json
import os
import time
import contextlib

import cv2
from ultralytics import YOLO
from moviepy import VideoFileClip  # Corrected import statement
import numpy as np

import supervision as sv
from tqdm import tqdm


class Detector:
    """
    Encapsulates the object detection logic with optional frame cropping.
    """

    def __init__(self, model_path, crop_params=None):
        self.model = YOLO(model_path)
        self.crop_params = crop_params or {'top': 0, 'bottom': 0, 'left': 0, 'right': 0}
        self.cropped_width = None
        self.cropped_height = None
        self.model_path = model_path

    def _preprocess_frame(self, frame):
        """Crops the frame based on initialized crop parameters."""
        height, width, _ = frame.shape
        top = self.crop_params['top']
        bottom = self.crop_params['bottom']
        left = self.crop_params['left']
        right = self.crop_params['right']

        self.cropped_height = height - top - bottom
        self.cropped_width = width - left - right

        return frame[top:height - bottom, left:width - right]

    def process_frame(self, frame):
        """
        Performs inference on a single frame.

        Args:
            frame (np.array): A single video frame.

        Returns:
            dict: A dictionary of detection results with adjusted coordinates.
        """
        original_height, original_width, _ = frame.shape
        cropped_frame = self._preprocess_frame(frame)

        with contextlib.redirect_stdout(open(os.devnull, 'w')):
            results = self.model(cropped_frame, verbose=False)

        inference_speed = results[0].speed

        detections = []
        for result in results[0].boxes:
            bbox_normalized_cropped = result.xywhn[0].cpu().numpy()
            conf = float(result.conf[0].cpu().numpy())
            cls = int(result.cls[0].cpu().numpy())

            x_center_cropped, y_center_cropped, w_cropped, h_cropped = bbox_normalized_cropped
            x1_cropped = int((x_center_cropped - w_cropped / 2) * self.cropped_width)
            y1_cropped = int((y_center_cropped - h_cropped / 2) * self.cropped_height)
            x2_cropped = int((x_center_cropped + w_cropped / 2) * self.cropped_width)
            y2_cropped = int((y_center_cropped + w_cropped / 2) * self.cropped_height)

            crop_left = self.crop_params['left']
            crop_top = self.crop_params['top']
            x1_original = x1_cropped + crop_left
            y1_original = y1_cropped + crop_top
            x2_original = x2_cropped + crop_left
            y2_original = y2_cropped + crop_top

            x_center_original_norm = (x1_original + x2_original) / (2 * original_width)
            y_center_original_norm = (y1_original + y2_original) / (2 * original_height)
            w_original_norm = (x2_original - x1_original) / original_width
            h_original_norm = (y2_original - y1_original) / original_height

            bbox_original_normalized = [x_center_original_norm, y_center_original_norm, w_original_norm,
                                        h_original_norm]

            detections.append({
                "bbox": bbox_original_normalized,
                "conf": conf,
                "cls": cls
            })

        return {
            "inference_speed": inference_speed,
            "boxes": detections,
            "original_frame_dims": [original_width, original_height]
        }


def _annotate_frame(frame, frame_data):
    """Draws bounding boxes and labels on a frame based on detection data."""
    annotated_frame = frame.copy()
    original_width, original_height = frame_data.get("original_frame_dims")

    for box_data in frame_data["boxes"]:
        bbox_norm = box_data["bbox"]
        conf = box_data["conf"]
        cls = box_data["cls"]

        x_center_norm, y_center_norm, w_norm, h_norm = bbox_norm
        x1 = int((x_center_norm - w_norm / 2) * original_width)
        y1 = int((y_center_norm - h_norm / 2) * original_height)
        x2 = int((x_center_norm + w_norm / 2) * original_width)
        y2 = int((y_center_norm + h_norm / 2) * original_height)

        color = (0, 255, 0)
        thickness = 2
        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, thickness)
        label = f"Class {cls}: {conf:.2f}"
        cv2.putText(annotated_frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    return annotated_frame


def process_video(video_path, detector, save_json=True, save_annotated_video=False, output_video_path=None,
                  json_path=None):
    """
    Processes a video file frame by frame and returns predictions and loaded frames.
    """
    if save_annotated_video and not output_video_path:
        raise ValueError("output_video_path must be specified when save_annotated_video is True.")
    if save_json and not json_path:
        raise ValueError("json_path must be specified when save_json is True.")

    video_info = sv.VideoInfo.from_video_path(video_path)
    frame_generator = sv.get_video_frames_generator(video_path)

    predictions = []
    frames = []

    if save_annotated_video:
        sink = sv.VideoSink(output_video_path, video_info=video_info)
    else:
        sink = contextlib.nullcontext()

    with sink:
        for frame_idx, frame in enumerate(
                tqdm(frame_generator, total=video_info.total_frames, desc="Processing Video")):
            frame_data = detector.process_frame(frame)
            frame_data["frame_count"] = frame_idx
            predictions.append(frame_data)
            frames.append(frame)  # Store the original frame

            if save_annotated_video:
                annotated_frame = _annotate_frame(frame, frame_data)
                sink.write_frame(annotated_frame)

    if save_json:
        with open(json_path, "w") as f:
            json.dump(predictions, f, indent=4)
            print(f"Predictions saved to {json_path}")

    return predictions, frames


def process_image_folder(folder_path, detector, save_json=True, save_annotated_images=False, output_folder=None,
                         json_path=None):
    """
    Processes all images in a folder and returns predictions and loaded frames.
    """
    if save_annotated_images and not output_folder:
        raise ValueError("output_folder must be specified when save_annotated_images is True.")
    if save_json and not json_path:
        raise ValueError("json_path must be specified when save_json is True.")

    image_files = sorted([os.path.join(folder_path, f) for f in os.listdir(folder_path) if
                          f.lower().endswith(('.png', '.jpg', '.jpeg'))])

    predictions = []
    frames = []

    for image_path in tqdm(image_files, desc="Processing Images"):
        frame = cv2.imread(image_path)
        if frame is None:
            continue

        image_data = detector.process_frame(frame)
        image_data["filename"] = os.path.basename(image_path)
        predictions.append(image_data)
        frames.append(frame)  # Store the original frame

        if save_annotated_images:
            annotated_frame = _annotate_frame(frame, image_data)
            output_image_path = os.path.join(output_folder, os.path.basename(image_path))
            cv2.imwrite(output_image_path, annotated_frame)

    if save_json:
        with open(json_path, "w") as f:
            json.dump(predictions, f, indent=4)
            print(f"Predictions saved to {json_path}")

    return predictions, frames


def process_stream(stream_source, detector):
    """
    Processes a video stream frame by frame.

    Args:
        stream_source (int or str): The source of the stream (e.g., 0 for a webcam, a URL for a network stream).
        detector (Detector): An instance of the Detector class.

    Returns:
        tuple: A tuple containing a list of predictions and a list of frames.
    """
    cap = cv2.VideoCapture(stream_source)
    if not cap.isOpened():
        print("Error: Could not open video stream.")
        return [], []

    predictions = []
    frames = []
    frame_idx = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                # Break the loop if reading a frame fails (end of stream or error)
                break

            # Process the frame using the detector
            frame_data = detector.process_frame(frame)
            frame_data["frame_count"] = frame_idx
            predictions.append(frame_data)
            frames.append(frame)  # Store the original frame

            # Optional: Display the frame in a window
            annotated_frame = _annotate_frame(frame, frame_data)
            cv2.imshow('Live Stream', annotated_frame)

            # Break the loop if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            frame_idx += 1

    finally:
        cap.release()
        cv2.destroyAllWindows()

    return predictions, frames


if __name__ == "__main__":
    # Define paths
    model_path = r"C:\Users\chris\foosball-statistics\weights\yolov11n_imgsz640_Topview.pt"
    video_path = r"C:\Users\chris\foosball-statistics\foosball-videos\SideKick Harburg - SiMo Doppel  05.08.2025.mp4"
    json_path = "output_predictions.json"
    annotated_video_path = "output_video_annotated.mp4"

    # Define crop parameters
    crop_params = {'left': 300, 'right': 300, 'top': 0, 'bottom': 0}

    # Initialize the detector
    detector = Detector(model_path, crop_params)

    start_time = time.time()

    # Process the video
    predictions_list, loaded_frames = process_video(
        video_path=video_path,
        detector=detector,
        save_json=True,
        save_annotated_video=True,
        output_video_path=annotated_video_path,
        json_path=json_path
    )

    end_time = time.time()
    print(f"Video processing completed in {(end_time - start_time):.2f} seconds.")
    print(f"The total number of frames processed is: {len(loaded_frames)}")

    # --- Create the final video with audio (Optional, and only if annotated video was created) ---
    if os.path.exists(annotated_video_path):
        try:
            processed_video = VideoFileClip(annotated_video_path)
            original_video = VideoFileClip(video_path)
            final_output_video_path = "final_with_audio.mp4"

            processed_video = processed_video.with_audio(original_video.audio)
            processed_video.write_videofile(final_output_video_path, codec='libx264', audio_codec='aac')
            print(f"Final video with audio saved to {final_output_video_path}")
        except Exception as e:
            print(f"Error creating final video with audio: {e}")
