import cv2
import os


def extract_images_from_videos(input_folder, output_folder, total_images=1000):
    # Create output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Get list of all video files in the input folder
    video_files = [f for f in os.listdir(input_folder) if f.endswith('.mp4')]
    if not video_files:
        print("No video files found in the input folder.")
        return

    # Calculate the total number of frames from all videos
    total_frames = 0
    video_cap_list = []
    for video_file in video_files:
        video_path = os.path.join(input_folder, video_file)
        cap = cv2.VideoCapture(video_path)
        total_frames += int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        video_cap_list.append(cap)

    # Calculate the interval to extract frames
    frame_interval = total_frames // total_images

    # Loop through the videos and extract frames
    extracted_images = 0
    current_frame = 0
    for cap, video_file in zip(video_cap_list, video_files):
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            # Only save the frame if the current frame matches the interval
            if current_frame % frame_interval == 0 and extracted_images < total_images:
                image_name = f"difficult_images_{extracted_images + 1}.jpg"
                image_path = os.path.join(output_folder, image_name)
                cv2.imwrite(image_path, frame)
                extracted_images += 1
            current_frame += 1
            if extracted_images >= total_images:
                break
        cap.release()

    print(f"Extraction complete. {extracted_images} images saved to {output_folder}.")

# Example usage:
extract_images_from_videos(r"D:\Difficult Sequences", r"D:\foosball_datasets\difficult_images")
