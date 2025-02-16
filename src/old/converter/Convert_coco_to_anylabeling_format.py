import json
import os
import shutil


def coco_to_anylabeling(input_dir, output_dir):
    # Ensure output directories exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for dataset_type in ['train', 'valid', 'test']:
        input_annotations_path = os.path.join(input_dir, dataset_type, "_annotations.coco.json")
        input_images_path = os.path.join(input_dir, dataset_type)

        output_dataset_path = os.path.join(output_dir, dataset_type)
        if not os.path.exists(output_dataset_path):
            os.makedirs(output_dataset_path)

        # Read COCO annotations
        with open(input_annotations_path, 'r') as f:
            coco_data = json.load(f)

        # Organize image annotations
        image_annotations = {}
        for annotation in coco_data["annotations"]:
            image_id = annotation["image_id"]
            if image_id not in image_annotations:
                image_annotations[image_id] = []
            image_annotations[image_id].append(annotation)

        for image in coco_data["images"]:
            image_id = image["id"]
            annotations = image_annotations.get(image_id, [])

            anylabeling_data = {
                "version": "0.3.3",
                "flags": {},
                "shapes": [],
                "imagePath": image["file_name"],
                "imageData": None,
                "imageHeight": image["height"],
                "imageWidth": image["width"]
            }

            for annotation in annotations:
                label = next(cat["name"] for cat in coco_data["categories"] if cat["id"] == annotation["category_id"])
                x, y, width, height = annotation["bbox"]
                shape_data = {
                    "label": label,
                    "text": "",
                    "points": [
                        [x, y],
                        [x + width, y + height]
                    ],
                    "group_id": None,
                    "shape_type": "rectangle",
                    "flags": {}
                }
                anylabeling_data["shapes"].append(shape_data)

            # Save the converted data for each image
            output_file_path = os.path.join(output_dataset_path, os.path.splitext(image["file_name"])[0] + ".json")
            with open(output_file_path, 'w') as f:
                json.dump(anylabeling_data, f, indent=4)

            # Copy the image file
            shutil.copy(os.path.join(input_images_path, image["file_name"]), output_dataset_path)

        # Save meta-data above the dataset folders for later conversion
        metadata = {
            "info": coco_data["info"],
            "licenses": coco_data["licenses"],
            "categories": coco_data["categories"]
        }
        with open(os.path.join(output_dir, f"{dataset_type}_metadata.json"), 'w') as f:
            json.dump(metadata, f, indent=4)


# Example usage:
coco_dir = r"C:\Users\chris\foosball-dataset\big foosball dataset adjusted"
anylabeling_dir = r"C:\Users\chris\foosball-dataset\big foosball dataset adjusted anylabeling format"

coco_to_anylabeling(coco_dir, anylabeling_dir)
