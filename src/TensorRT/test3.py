from ultralytics import YOLO

# model = YOLO(r"C:\Users\chris\foosball-statistics\runs\detect\train2\weights\best.pt")
# model.export(format="engine")  # creates 'yolov8n.engine'

# Run inference
# model = YOLO(r"C:\Users\chris\foosball-statistics\runs\detect\train2\weights\best.pt")
# results = model.predict("https://ultralytics.com/images/bus.jpg")


import os
# Set the CUDA_MODULE_LOADING environment variable to LAZY
from ultralytics import YOLO
os.environ['CUDA_MODULE_LOADING'] = 'LAZY'
model = YOLO(r"/runs/detect/train5/weights/best_run5.pt")
# model.export(
#     format="engine",
#     dynamic=True,
#     batch=8,
#     workspace=4,
#     int8=True,
#     data=r"D:\New_Big_Foosball_Dataset\Second_Prototype_Dataset\dataset.yaml"
# )

model.export(
    format="engine",
    dynamic=True,
    batch=8,
    workspace=4,
    half=True
)


# Load the exported TensorRT INT8 model
# model = YOLO("yolov8n.engine", task="detect")

# Run inference
# result = model.predict("https://ultralytics.com/images/bus.jpg")





# import os
# import shutil
#
#
# def replace_files(folder_a, folder_b):
#     # Traverse folder A and find all .txt files
#     for root_a, dirs_a, files_a in os.walk(folder_a):
#         for file_a in files_a:
#             if file_a.endswith('.txt'):
#                 # Construct the full path to the file in folder A
#                 file_a_path = os.path.join(root_a, file_a)
#
#                 # Look for the same file in folder B
#                 for root_b, dirs_b, files_b in os.walk(folder_b):
#                     if file_a in files_b:
#                         # Construct the full path to the file in folder B
#                         file_b_path = os.path.join(root_b, file_a)
#
#                         # Replace the file in folder B with the one from folder A
#                         shutil.copy2(file_a_path, file_b_path)
#                         print(f"Replaced: {file_b_path} with {file_a_path}")
#
#
# # Example usage
# folder_a = r"D:\New_Big_Foosball_Dataset\labels_in_one_folder_hard\new_labels"  # Replace with the path to folder A
# folder_b = r"D:\New_Big_Foosball_Dataset\folders"  # Replace with the path to folder B
#
# replace_files(folder_a, folder_b)


# ffmpeg -i C:\Users\chris\foosball-statistics\Second_Prototype_without_Impossible_Bundesliga_2024_Bamberg_Kicker_Crew_Bonn.mp4 -vf scale=640:360 -b:v 1000k C:\Users\chris\foosball-statistics\Second_Prototype_without_Impossible_Bundesliga_2024_Bamberg_Kicker_Crew_Bonn_reduced.mp4