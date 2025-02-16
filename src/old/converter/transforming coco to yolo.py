import os
import json
from pathlib import Path
import shutil


def convert_coco_to_yolo_annotation(coco_ann, img_width, img_height):
    x_center = (coco_ann['bbox'][0] + coco_ann['bbox'][2] / 2) / img_width
    y_center = (coco_ann['bbox'][1] + coco_ann['bbox'][3] / 2) / img_height
    width = coco_ann['bbox'][2] / img_width
    height = coco_ann['bbox'][3] / img_height
    return f"{coco_ann['category_id']} {x_center} {y_center} {width} {height}"


def main(coco_root, yolo_root):
    coco_to_yolo_splits = {
        "train": "train",
        "valid": "val",
        "test": "test"
    }

    file_counter = 0  # Counter for renaming

    for coco_split, yolo_split in coco_to_yolo_splits.items():
        coco_ann_path = Path(coco_root) / coco_split / f"_annotations.coco.json"
        with open(coco_ann_path, "r") as f:
            data = json.load(f)

        img_id_to_size = {}
        img_id_to_filename = {}
        for img in data['images']:
            img_id_to_size[img['id']] = (img['width'], img['height'])
            img_id_to_filename[img['id']] = Path(img['file_name']).stem  # Get the filename without extension

        # Group annotations by image
        annotations_grouped = {}
        for ann in data['annotations']:
            img_id = ann['image_id']
            if img_id not in annotations_grouped:
                annotations_grouped[img_id] = []
            annotations_grouped[img_id].append(ann)

        # Copying images and saving annotations
        for img_id, ann_list in annotations_grouped.items():
            img_width, img_height = img_id_to_size[img_id]
            yolo_annotations = [convert_coco_to_yolo_annotation(ann, img_width, img_height) for ann in ann_list]

            # Copying image
            original_filename = img_id_to_filename[img_id]
            trimmed_filename = original_filename.split(".")[0]  # Keep everything before the first dot

            # Use the counter to generate new filename
            new_filename = f"{trimmed_filename}.{str(file_counter).zfill(5)}"

            source_img_path = Path(coco_root) / coco_split / f"{original_filename}.jpg"
            destination_img_path = Path(yolo_root) / "images" / yolo_split / f"{new_filename}.jpg"
            shutil.copy(source_img_path, destination_img_path)

            # Writing annotations
            out_path = Path(yolo_root) / "labels" / yolo_split / f"{new_filename}.txt"
            with open(out_path, "w") as out_file:
                out_file.write("\n".join(yolo_annotations))

            file_counter += 1  # Increment the counter

    # Create the yaml file if not created already
    if not os.path.exists(Path(yolo_root) / "dataset.yaml"):
        with open(Path(yolo_root) / "dataset.yaml", "w", encoding="utf-8") as yaml_file:
            category_names = {cat['id']: cat['name'] for cat in data['categories']}
            yaml_file.write("# Ultralytics YOLO 🚀\n")

            corrected_path = yolo_root.replace('\\', '/')
            yaml_file.write(f"path: {corrected_path}  # dataset root dir\n")

            yaml_file.write("train: images/train  # train images\n")
            yaml_file.write("val: images/val  # val images\n")
            yaml_file.write("test: images/test  # test images (optional)\n\n")
            yaml_file.write("# Classes\nnames:\n")

            for cat_id, cat_name in category_names.items():
                yaml_file.write(f"  {cat_id}: {cat_name}\n")


if __name__ == "__main__":
    COCO_ROOT = r"C:\Users\chris\foosball-dataset\big foosball dataset adjusted"
    YOLO_ROOT = r"C:\Users\chris\foosball-dataset\big foosball dataset adjusted yolo format"

    # Ensure the images and labels directories exist
    for split in ["train", "val", "test"]:
        os.makedirs(os.path.join(YOLO_ROOT, "images", split), exist_ok=True)
        os.makedirs(os.path.join(YOLO_ROOT, "labels", split), exist_ok=True)

    main(COCO_ROOT, YOLO_ROOT)