import cv2
from ultralytics import YOLO

# Load the YOLOv11 model
model = YOLO("/home/freystec/foosball-statistics/weights/ai_foosball_yolo11n_KD_epoch24.pt")  # Ensure you have the correct model file

# Open video file
input_video_path = "/home/freystec/foosball-statistics/foosball-videos/Leonhart_clip.mp4"
output_video_path = "yolov11m_Leonhart_Topview_Test_video_destilled.mp4"
cap = cv2.VideoCapture(input_video_path)

# Get video properties
fps = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Make YOLO predictions
    results = model(frame)

    # Draw bounding boxes on the frame
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = box.conf[0].item()
            label = result.names[int(box.cls[0].item())]

            # Draw rectangle and label
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"{label} {conf:.2f}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Write the frame to the output video
    out.write(frame)

# Release resources
cap.release()
out.release()
cv2.destroyAllWindows()
print("Processing complete. Output saved to", output_video_path)
