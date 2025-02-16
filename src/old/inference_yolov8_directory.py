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
model = YOLO(r"C:\Users\chris\Foosball Detector\runs\detect\train8\weights\best.pt")

# Define path to directory containing images and videos for inference
source = r"D:\test_data\images\3"

# Run inference on the source
results = model(source, stream=True)  # generator of Results objects


# Visualize the resultsx
for i, r in enumerate(results):
    # Plot results image
    im_bgr = r.plot()  # BGR-order numpy array
    im_rgb = Image.fromarray(im_bgr[..., ::-1])  # RGB-order PIL image
    im_rgb.show()
    # Show results to screen (in supported environments)

