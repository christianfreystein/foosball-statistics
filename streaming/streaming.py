# from ultralytics import YOLO
#
# model = YOLO(r"C:\Users\chris\foosball-statistics\runs\detect\train5\weights\best.pt")
#
# source = 'rtmp://nukular.wtf/kickercrew'
#
# result = model.predict(source, save=True, conf=0.5)
# print(result)
# print("boxes:")
# for box in result[0].boxes:
#     print(box)


import cv2
from ultralytics import YOLO

# Initialize the YOLO model with the path to the weights
model = YOLO(r"C:\Users\chris\foosball-statistics\runs\detect\train5\weights\best.pt")

# Define the source, can be an RTMP stream or a local video
source = 'rtmp://nukular.wtf/kickercrew'

# Initialize video capture for the stream
cap = cv2.VideoCapture(source)

frame_count = 0  # Initialize a frame counter
result = None    # Initialize a variable to store the previous frame's result

while True:
    ret, frame = cap.read()  # Read a frame from the stream
    if not ret:
        print("Failed to grab frame from stream")
        break

    frame_count += 1

    # Only perform YOLO inference on every second frame
    if frame_count % 2 == 0:
        result = model.predict(frame, conf=0.5)

    # If we have a detection result, use it to annotate the frame
    if result:
        annotated_frame = result[0].plot()  # Draw bounding boxes
    else:
        annotated_frame = frame  # If no result, show the original frame

    # Display the frame with or without annotations
    cv2.imshow("YOLO Stream", annotated_frame)

    # Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
