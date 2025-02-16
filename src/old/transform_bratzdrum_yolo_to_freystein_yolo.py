import os
import shutil
from tqdm import tqdm


def create_dir_structure(base_path):
    # Create the target folder structure
    new_structure = [
        "images/train", "images/val", "images/test",
        "labels/train", "labels/val", "labels/test"
    ]
    for folder in new_structure:
        os.makedirs(os.path.join(base_path, folder), exist_ok=True)


def copy_files(src_folder, dest_folder, file_type):
    # Copy files from source to destination folder
    if os.path.exists(src_folder):
        files = [file for file in os.listdir(src_folder) if file.endswith(file_type)]
        for file_name in tqdm(files, desc=f"Copying {file_type} files from {src_folder} to {dest_folder}"):
            shutil.copy(os.path.join(src_folder, file_name), os.path.join(dest_folder, file_name))
    return files


def create_empty_label_files(image_files, label_folder):
    # Create empty label files if they are missing
    for image_file in image_files:
        base_name = os.path.splitext(image_file)[0]
        label_file = base_name + ".txt"
        label_file_path = os.path.join(label_folder, label_file)
        if not os.path.exists(label_file_path):
            with open(label_file_path, 'w') as f:
                pass


def transform_dataset(base_path, new_base_path):
    # Create the new directory structure
    create_dir_structure(new_base_path)

    # Define the original and new directories
    original_folders = ["train", "val", "test"]

    for folder in original_folders:
        src_img_folder = os.path.join(base_path, folder, "images")
        src_lbl_folder = os.path.join(base_path, folder, "labels")

        dest_img_folder = os.path.join(new_base_path, "images", folder)
        dest_lbl_folder = os.path.join(new_base_path, "labels", folder)

        image_files = copy_files(src_img_folder, dest_img_folder, ".jpg")
        copy_files(src_lbl_folder, dest_lbl_folder, ".txt")

        # Create empty label files if missing
        create_empty_label_files(image_files, dest_lbl_folder)


if __name__ == "__main__":
    base_path = r"D:\foosball_datasets\big_dataset_bratzdrum" # Original dataset folder path
    new_base_path = r"D:\foosball_datasets\big_dataset_bratzdrum_yolo_adjusted"  # New dataset folder path
    transform_dataset(base_path, new_base_path)




