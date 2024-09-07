import os
import shutil

# Define the paths to the folders
txt_folder = r"D:\foosball_datasets\big foosball dataset adjusted yolo format reduced\labels\train"
jpg_folder = r"D:\foosball_datasets\big foosball dataset adjusted yolo format reduced\images\train"
destination_folder = r"D:\labels\big_foosball_dataset_train\obj_train_data"

# Create the destination folder if it doesn't exist
if not os.path.exists(destination_folder):
    os.makedirs(destination_folder)

# Get a list of all txt and jpg filenames (without extensions)
txt_files = set(f.split('.txt')[0] for f in os.listdir(txt_folder) if f.endswith('.txt'))
jpg_files = set(f.split('.jpg')[0] for f in os.listdir(jpg_folder) if f.endswith('.jpg'))

# Find txt files that have a corresponding jpg file
txt_with_jpg = txt_files.intersection(jpg_files)

# Copy the matching txt files to the destination folder
for txt_file in txt_with_jpg:
    source_path = os.path.join(txt_folder, f"{txt_file}.txt")
    destination_path = os.path.join(destination_folder, f"{txt_file}.txt")
    shutil.copy2(source_path, destination_path)

print("Copying complete.")
