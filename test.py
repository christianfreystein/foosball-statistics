# import os
#
#
# def manage_files(images_dir, labels_dir):
#     # Get the list of image files and label files
#     image_files = set(f[:-4] for f in os.listdir(images_dir) if f.endswith('.jpg'))
#     label_files = set(f[:-4] for f in os.listdir(labels_dir) if f.endswith('.txt'))
#
#     # Find images without corresponding labels and labels without corresponding images
#     images_without_labels = image_files - label_files
#     labels_without_images = label_files - image_files
#
#     # Create empty TXT files for images without labels
#     for image in images_without_labels:
#         empty_txt_path = os.path.join(labels_dir, image + '.txt')
#         with open(empty_txt_path, 'w') as f:
#             pass  # Create an empty file
#
#     # Delete label files without corresponding images
#     for label in labels_without_images:
#         txt_path = os.path.join(labels_dir, label + '.txt')
#         os.remove(txt_path)
#
#     return images_without_labels, labels_without_images
#
#
# # Paths to the images and labels directories
# images_dir = r"D:\foosball_datasets\big_dataset_bratzdrum\train\images"
# labels_dir = r"D:\foosball_datasets\big_dataset_bratzdrum\train\labels"
#
# images_without_labels, labels_without_images = manage_files(images_dir, labels_dir)
#
# print("Images without corresponding labels (created empty txt files):")
# for image in images_without_labels:
#     print(image + '.jpg')
#
# print("\nLabels without corresponding images (deleted these txt files):")
# for label in labels_without_images:
#     print(label + '.txt')

import os
import shutil

# Define the paths
source_base_path = r"D:\foosball_datasets\big_dataset_bratzdrum_freystein_bonn"
destination_path = r"D:\foosball_datasets\big_dataset_bratzdrum_bonn_combined"

# Ensure the destination folders exist
os.makedirs(os.path.join(destination_path, 'images', 'train'), exist_ok=True)
os.makedirs(os.path.join(destination_path, 'images', 'val'), exist_ok=True)
os.makedirs(os.path.join(destination_path, 'images', 'test'), exist_ok=True)
os.makedirs(os.path.join(destination_path, 'labels', 'train'), exist_ok=True)
os.makedirs(os.path.join(destination_path, 'labels', 'val'), exist_ok=True)
os.makedirs(os.path.join(destination_path, 'labels', 'test'), exist_ok=True)


# Function to copy files from source to destination
def copy_files(source, dest, filter_func=None):
    for root, dirs, files in os.walk(source):
        for file in files:
            source_file = os.path.join(root, file)
            relative_path = os.path.relpath(source_file, source)
            destination_file = os.path.join(dest, relative_path)

            os.makedirs(os.path.dirname(destination_file), exist_ok=True)

            if filter_func and file.endswith('.txt'):
                with open(source_file, 'r') as f:
                    lines = f.readlines()
                filtered_lines = filter_func(lines)
                with open(destination_file, 'w') as f:
                    f.writelines(filtered_lines)
            else:
                shutil.copy2(source_file, destination_file)


# Function to filter lines where the class is 0
def filter_class_0(lines):
    return [line for line in lines if line.startswith('0 ')]


# Iterate over each dataset folder in the source base path
for dataset_folder in os.listdir(source_base_path):
    dataset_path = os.path.join(source_base_path, dataset_folder)
    if os.path.isdir(dataset_path):
        # Copy images and labels with filtering for txt files
        for subfolder in ['images/train', 'images/val', 'images/test', 'labels/train', 'labels/val', 'labels/test']:
            source_subfolder_path = os.path.join(dataset_path, subfolder)
            destination_subfolder_path = os.path.join(destination_path, subfolder)
            if 'labels' in subfolder:
                copy_files(source_subfolder_path, destination_subfolder_path, filter_func=filter_class_0)
            else:
                copy_files(source_subfolder_path, destination_subfolder_path)

print("Datasets copied and filtered successfully.")



