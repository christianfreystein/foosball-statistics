import cv2
import random

# Define the path to the video file
video_path = r"/Second_Prototype_without_Impossible_Westermann_Bade_Hoffmann_Spredeman.mp4"

# Open the video file
cap = cv2.VideoCapture(video_path)

# Get total number of frames
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

if frame_count > 0:
    # Select a random frame index
    random_frame_index = random.randint(0, frame_count - 1)

    # Set the video to the random frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, random_frame_index)

    # Read the frame
    success, frame = cap.read()

    if success:
        # Define the output image path
        output_image_path = r"C:\Users\chris\foosball-statistics\random_frame.jpg"

        # Save the frame as an image
        cv2.imwrite(output_image_path, frame)
        print(f"Random frame saved to {output_image_path}")
    else:
        print("Failed to capture the frame.")

else:
    print("Error: Could not retrieve frame count.")

# Release the video capture
cap.release()



