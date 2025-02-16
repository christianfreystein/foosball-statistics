from ultralytics import YOLO

# Load the YOLOv8 model
# model = YOLO(r"C:\Users\chris\Foosball Detector\weights\yolov9s_foosdetect.pt")
model = YOLO(r"C:\Users\chris\Foosball Detector\runs\detect\train20\weights\best.pt")
model.export(format="engine")
