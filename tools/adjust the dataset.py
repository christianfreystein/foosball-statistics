import json
import os
import random
from albumentations import Resize, Compose
import cv2
import shutil
from tqdm import tqdm

# Root path to the dataset
ROOT_DATASET_PATH = r"C:\Users\chris\foosball-dataset\big foosball dataset"

# Path to the new dataset directory
NEW_DATASET_PATH = r"C:\Users\chris\foosball-dataset\big foosball dataset adjusted"

# Ensure new dataset directory exists
os.makedirs(NEW_DATASET_PATH, exist_ok=True)


# Used for generating new unique image IDs
def merge_coco_data(root_dataset_path):
    """
    Merge COCO dataset from train, valid, and test splits.

    Parameters:
    - root_dataset_path (str): Root path to the dataset.

    Returns:
    - merged_data (dict): The merged COCO dataset.
    - absolute_paths (dict): Dictionary mapping image filenames to their absolute paths.
    """
    merged_data = {
        "images": [],
        "annotations": [],
        "categories": [],
        "info": {},
        "licenses": []
    }
    absolute_paths = {}
    current_image_id = 0

    for folder in tqdm(["train", "valid", "test"], desc="merging cocodata", unit="folder"):
        with open(os.path.join(root_dataset_path, folder, "_annotations.coco.json"), "r") as f:
            data = json.load(f)

            # Adjusting image IDs and related annotations
            id_map = {}
            for img in data["images"]:
                old_id = img["id"]
                img["id"] = current_image_id
                id_map[old_id] = current_image_id
                current_image_id += 1

            for ann in data["annotations"]:
                ann["image_id"] = id_map[ann["image_id"]]

            merged_data["images"].extend(data["images"])
            merged_data["annotations"].extend(data["annotations"])
            if "categories" in data and not merged_data["categories"]:
                merged_data["categories"] = data["categories"]
                merged_data["info"] = data["info"]
                merged_data["licenses"] = data["licenses"]

            # Populate the absolute_paths dictionary
            for image in data["images"]:
                absolute_paths[image["id"]] = os.path.join(root_dataset_path, folder, image["file_name"])

    return merged_data, absolute_paths


merged_data, abs_paths = merge_coco_data(ROOT_DATASET_PATH)


# # Step 1: Adjust image sizes and bounding boxes for images with "frame" in their name.
# transform = Compose([Resize(width=640, height=480)], bbox_params={'format': 'coco', 'label_fields': ['category_ids']})
#
# for image in merged_data["images"]:
#     if "frame" in image["file_name"] and image["width"] == 640 and image["height"] == 640:
#         # Use the dictionary to get the image path
#         img_path = absolute_paths[image["file_name"]]
#         img = cv2.imread(img_path)
#         img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#
#         image_annotations = [ann for ann in merged_data["annotations"] if ann["image_id"] == image["id"]]
#         bboxes = [ann["bbox"] for ann in image_annotations]
#         category_ids = [ann["category_id"] for ann in image_annotations]
#
#         transformed = transform(image=img, bboxes=bboxes, category_ids=category_ids)
#
#         # Save the transformed image
#         transformed_img = transformed["image"]
#         cv2.imwrite(img_path, cv2.cvtColor(transformed_img, cv2.COLOR_RGB2BGR))
#
#         # Update image dimensions
#         image["height"] = 480
#
#         # Update bounding boxes
#         for ann, bbox in zip(image_annotations, transformed["bboxes"]):
#             ann["bbox"] = bbox

def generate_new_filenames(data, file_counter):
    """
    Generate new filenames for the images in the dataset.

    Returns:
    - Dictionary mapping image_ids to new filenames.
    """
    id_to_new_filename = {}

    for img in data["images"]:
        original_filename = img["file_name"]
        trimmed_filename = original_filename.split(".")[0]  # Keep everything before the first dot

        # Use the counter to generate new filename
        new_filename = f"{trimmed_filename}.{str(file_counter).zfill(5)}.jpg"
        id_to_new_filename[img["id"]] = new_filename

        file_counter += 1

    return id_to_new_filename


def adjust_annotations(data, id_to_new_filename):
    """
    Adjust annotations based on the new filenames.
    """
    for img in data:
        img["file_name"] = id_to_new_filename[img["id"]]


# Step 2: Remove 3/4 of the images with "video_mp4" in their name.
images_to_remove = [image for image in merged_data["images"] if "video_mp4" in image["file_name"]]
random.shuffle(images_to_remove)
images_to_remove = images_to_remove[:int(0.75 * len(images_to_remove))]

for img in images_to_remove:
    merged_data["images"].remove(img)
    merged_data["annotations"] = [ann for ann in merged_data["annotations"] if ann["image_id"] != img["id"]]

file_counter = 0

# Generate new filenames for each image
id_to_new_filename = generate_new_filenames(merged_data, file_counter)

# Step 3: Make a new train, valid and test split.
random.shuffle(merged_data["images"])
total = len(merged_data["images"])
train_split = merged_data["images"][:int(0.7 * total)]
valid_split = merged_data["images"][int(0.7 * total):int(0.9 * total)]
test_split = merged_data["images"][int(0.9 * total):]

splits = {
    "train": train_split,
    "valid": valid_split,
    "test": test_split
}


def copy_image_to_split_folder(image_id, abs_path, id_to_new_filename, new_dataset_path, split_name):
    """
    Copy an image to the corresponding split folder (train, valid, test) in the new dataset with a new filename.
    """
    source_path = abs_path[image_id]
    destination_dir = os.path.join(new_dataset_path, split_name)
    os.makedirs(destination_dir, exist_ok=True)

    destination_path = os.path.join(destination_dir, id_to_new_filename[image_id])
    shutil.copy(source_path, destination_path)


# Handle each split
for split_name, split_data in splits.items():

    # Copy images to the new dataset folder based on the split using new filenames
    for image in tqdm(split_data, desc="copy images", unit="image"):
        copy_image_to_split_folder(image["id"], abs_paths, id_to_new_filename, NEW_DATASET_PATH, split_name)

    # Adjust annotations with the new filenames
    adjust_annotations(split_data, id_to_new_filename)

    # Save the split annotation file
    split_dir = os.path.join(NEW_DATASET_PATH, split_name)
    with open(os.path.join(split_dir, "_annotations.coco.json"), "w") as f:
        json_data = {
            "info": merged_data.get("info", {}),
            "licenses": merged_data.get("licenses", []),
            "categories": merged_data["categories"],
            "images": split_data,
            "annotations": [ann for ann in merged_data["annotations"] if
                            ann["image_id"] in [img["id"] for img in split_data]]
        }
        json.dump(json_data, f)
