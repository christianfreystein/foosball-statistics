import os
import shutil
import glob
import random
from tqdm import tqdm

# Define the labels to merge and ignore
merge_labels = {
    '0': '0',  # Ball (slow)
    '1': '0',  # Ball (slow)
    '2': '0',  # Ball (slow)
    '6': '0',  # Ball (slow)
    '3': '1',  # Ball (fast)
    '4': '1'   # Ball (fast)
}
ignore_labels = ['5']  # Ball (unclear)

# Define the output folder
reduced_labels_folder = r"D:\reduced_labels"

# Ensure the output folder exists
os.makedirs(reduced_labels_folder, exist_ok=True)

# Define the input folder
input_folder = r"D:\labels"  # replace with your actual input folder

# Automatically find all the dataset folders in the input folder
dataset_folders = [os.path.join(input_folder, dir) for dir in os.listdir(input_folder) if os.path.isdir(os.path.join(input_folder, dir))]

# Iterate over the dataset folders
for dataset_folder in dataset_folders:
    # Check if the 'obj_train_data' folder exists in the dataset folder
    if 'obj_train_data' not in os.listdir(dataset_folder):
        continue

    # Get the list of txt files
    txt_files = os.listdir(os.path.join(dataset_folder, 'obj_train_data'))

    # Iterate over the txt files
    for txt_file in txt_files:
        with open(os.path.join(dataset_folder, 'obj_train_data', txt_file), 'r') as f:
            lines = f.readlines()

        # Filter out the ignored labels and map the labels to new classes
        lines = [f"{merge_labels[line.split()[0]]} {' '.join(line.split()[1:])}" for line in lines if line.split()[0] not in ignore_labels and line.split()[0] in merge_labels]

        # Skip if there are no remaining entries
        if not lines:
            continue

        # Write the new file to the output folder
        with open(os.path.join(reduced_labels_folder, txt_file), 'w') as f:
            f.write('\n'.join(lines))

# Define the root output folder
root_output_folder = r"D:\training_dataset_2"

# Define the output folders
output_folders = {
    'labels': {
        'train': os.path.join(root_output_folder, 'labels/train'),
        'val': os.path.join(root_output_folder, 'labels/val')
    },
    'images': {
        'train': os.path.join(root_output_folder, 'images/train'),
        'val': os.path.join(root_output_folder, 'images/val')
    }
}

# Ensure the output folders exist
for output_type in output_folders.values():
    for output_folder in output_type.values():
        os.makedirs(output_folder, exist_ok=True)

# Get a list of all the txt file paths in the labels folder
txt_files = glob.glob(os.path.join(reduced_labels_folder, '*.txt'))

# Randomly split the txt files into train and val sets
random.shuffle(txt_files)
split_index = int(len(txt_files) * 0.99)
train_files = txt_files[:split_index]
val_files = txt_files[split_index:]

# Define the dataset splits
splits = {
    'train': train_files,
    'val': val_files
}


# Get a list of all the jpg file paths in the images folder and its subfolders
images_folder = 'D:\\dataset_random_frames_from_highlight_videos\\images'
jpg_files = glob.glob(os.path.join(images_folder, '**/*.jpg'), recursive=True)

# Iterate over the splits
for split, files in splits.items():
    # Define the output folders for this split
    labels_folder = output_folders['labels'][split]
    images_folder = output_folders['images'][split]

    # Iterate over the files
    for file in tqdm(files):
        # Copy the txt file to the labels folder
        shutil.copy(file, labels_folder)

        # Find and copy the corresponding jpg file to the images folder
        base_name = os.path.splitext(os.path.basename(file))[0]
        for jpg_file in jpg_files:
            if os.path.splitext(os.path.basename(jpg_file))[0] == base_name:
                shutil.copy(jpg_file, images_folder)
                break
