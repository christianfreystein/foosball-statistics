from ultralytics import YOLO
from PIL import Image
import cv2
#
# model = YOLO(r"C:\Users\chris\Foosball Detector\runs\detect\train8\weights\best.pt")
# # accepts all formats - image/dir/Path/URL/video/PIL/ndarray. 0 for webcam
#
# # results = model.predict(source="https://www.youtube.com/watch?v=aAXxONJDB0A", show=False, show_labels=False, show_conf=False, save=True) # Display preds. Accepts all YOLO predict arguments
#
#
#
# # Define source as YouTube video URL
# source = "https://www.youtube.com/watch?v=aAXxONJDB0A"

import matplotlib.pyplot as plt


import cv2
from ultralytics import YOLO

# Load the YOLOv8 model
model = YOLO('yolov8n.pt')

# Open the video file
video_path = r"C:\Users\chris\Foosball Detector\Dramatic Look [y8Kyi0WNg40].mp4"
cap = cv2.VideoCapture(video_path)

# Loop through the video frames
while cap.isOpened():
    # Read a frame from the video
    success, frame = cap.read()

    if success:
        # Run YOLOv8 inference on the frame
        results = model(frame)

        # Visualize the results on the frame
        annotated_frame = results[0].plot()

        # Display the annotated frame
        cv2.imshow("YOLOv8 Inference", annotated_frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        # Break the loop if the end of the video is reached
        break

# Release the video capture object and close the display window
cap.release()
cv2.destroyAllWindows()