from ultralytics import YOLO
import os
from glob import glob

# Load a pretrained YOLOv8n model
model = YOLO(r"C:\Users\chris\Foosball Detector\runs\detect\train11\weights\best.pt")

# Root directory path
root_dir = r"D:\test_data"
images_dir = os.path.join(root_dir, "images")
annotations_dir = os.path.join(root_dir, "annotations")

# Iterate over each sub-folder in the images directory
for folder in os.listdir(images_dir):
    folder_path = os.path.join(images_dir, folder)

    # Create a corresponding folder in the annotations directory
    save_folder_path = os.path.join(annotations_dir, folder, "obj_train_data")
    os.makedirs(save_folder_path, exist_ok=True)

    # Initialize a list to store the paths of the txt files for train.txt
    train_txt_paths = []

    # Iterate over each image in the sub-folder
    for img_path in glob(os.path.join(folder_path, "*.jpg")):
        # Run inference on an image
        results = model(img_path, save_txt=False)

        # Define the save path for the txt file
        img_name = os.path.splitext(os.path.basename(img_path))[0]
        save_path = os.path.join(save_folder_path, img_name + '.txt')
        results[0].save_txt(save_path)

        # Add the path of the txt file to train_txt_paths
        train_txt_paths.append(f"data/obj_train_data/{img_name}.jpg")

        # Open the txt file and delete all entries which belong to the class 1
        with open(save_path, 'r') as f:
            lines = f.readlines()

        with open(save_path, 'w') as f:
            for line in lines:
                if int(line.split()[0]) != 1:
                    f.write(line)

    # Create obj.data
    with open(os.path.join(annotations_dir, folder, "obj.data"), 'w') as f:
        f.write("classes = 4\n")
        f.write("train = data/train.txt\n")
        f.write("names = data/obj.names\n")
        f.write("backup = backup/\n")

    # Create obj.names
    with open(os.path.join(annotations_dir, folder, "obj.names"), 'w') as f:
        f.write("Ball\n")
        f.write("Ball-Sonderfall")
        f.write("Puppe\n")
        f.write("Puppe-Sonderfall")

    # Create train.txt
    with open(os.path.join(annotations_dir, folder, "train.txt"), 'w') as f:
        for path in train_txt_paths:
            f.write(path + '\n')

