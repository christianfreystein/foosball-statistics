from ultralytics import YOLO
import os

# Define the paths
source_folder = r"/home/freystec/small_diverse_dataset_for_testing"
output_folder = r"/home/freystec/testing_Nikolais_model"
model_path = r"/home/freystec/foosball-statistics/weights/edge_detection_Nikolai.pt" # Replace with the actual path to your model

# Make sure the output directory exists
os.makedirs(output_folder, exist_ok=True)

# Load a pre-trained YOLO model
model = YOLO(model_path)

# Perform prediction on the folder of images
# The 'source' argument can be a directory path
# 'save=True' saves the annotated images
# 'project' specifies the main directory to save the results
# 'name' specifies the sub-directory within the project folder. 'exist_ok=True' prevents errors if the folder already exists.
results = model.predict(source=source_folder, save=True, project=output_folder, name="", exist_ok=True)

# The 'results' object contains the prediction details
print("Prediction completed. Annotated images are saved in the following directory:")
print(f"{output_folder}\n")
print("Here is a summary of the results:")

for result in results:
    if result.boxes:
        print(f"Image: {os.path.basename(result.path)}")
        print(f"  Detections: {len(result.boxes)}")
        for box in result.boxes:
            class_name = model.names[int(box.cls)]
            confidence = box.conf.item()
            print(f"    - Class: {class_name}, Confidence: {confidence:.2f}")