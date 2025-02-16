import cv2
import json

# File paths
data_path = r"/Second_Prototype_without_Impossible_Westermann_Bade_Hoffmann_Spredeman_with_ball_status.json"
input_video_path = r"/Westermann Bade VS Hoffmann Spredeman [1sBcrUBZxlY].mp4"
output_video_path = r"/Second_Prototype_without_Impossible_Westermann_Bade_Hoffmann_Spredeman_with_ball_status.mp4"
# Color mapping for ball status
status_color_mapping = {
    "Left 2": (0, 0, 255),   # Red
    "Right 0": (0, 255, 0),  # Green
    "Left 1": (0, 165, 255),  # Orange
    "Right 1": (0, 255, 255), # Yellow
    "Left 0": (255, 0, 255), # Magenta
    "Right 2": (255, 255, 0), # Cyan
    "Outside": (128, 128, 128) # Gray
}

# Load the video file
cap = cv2.VideoCapture(input_video_path)

# Check if the video opened successfully
if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

# Load the JSON data
with open(data_path, 'r') as f:
    frames_data = json.load(f)

# Define the codec and create a VideoWriter object to save the video
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

# Iterate over each frame and the corresponding data
frame_idx = 0
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    if frame_idx < len(frames_data):
        frame_data = frames_data[frame_idx]
        ball_location = frame_data["current_ball_location"]
        ball_status = frame_data["ball_status"]

        # Get the color for the current ball status
        color = status_color_mapping.get(ball_status, (255, 255, 255))  # Default to white if status is not found

        # If the ball is detected, draw the bounding box
        if ball_location != [0, 0, 0, 0]:
            x_center, y_center, box_width, box_height = ball_location
            x_center_px = x_center * width
            y_center_px = y_center * height
            box_width_px = box_width * width
            box_height_px = box_height * height

            x1 = int(x_center_px - box_width_px / 2)
            y1 = int(y_center_px - box_height_px / 2)
            x2 = int(x_center_px + box_width_px / 2)
            y2 = int(y_center_px + box_width_px / 2)

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            # Print the coordinates of the center of the bounding box
            cv2.putText(frame, f"Center: ({x_center_px:.0f}, {y_center_px:.0f})", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2, cv2.LINE_AA)

        # Put the ball status text on the frame
        cv2.putText(frame, f"Status: {ball_status}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2, cv2.LINE_AA)

    # Write the frame with annotations
    out.write(frame)

    frame_idx += 1

# Release everything if job is finished
cap.release()
out.release()
cv2.destroyAllWindows()



