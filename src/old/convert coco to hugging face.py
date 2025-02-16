import json
import os
from collections import defaultdict

def convert_annotations(coco_annotation_path, output_annotation_path):
    with open(coco_annotation_path, 'r') as file:
        coco_annotations = json.load(file)

    new_annotations = defaultdict(lambda: {'file_name': '', 'objects': {'bbox': [], 'categories': []}})

    for annotation in coco_annotations['annotations']:
        image_id = annotation['image_id']
        image_info = next(img for img in coco_annotations['images'] if img['id'] == image_id)
        file_name = image_info['file_name']
        bbox = annotation['bbox']
        category_id = annotation['category_id']

        new_annotations[image_id]['file_name'] = file_name
        new_annotations[image_id]['objects']['bbox'].append(bbox)
        new_annotations[image_id]['objects']['categories'].append(category_id)

    with open(output_annotation_path, 'w') as file:
        for annotation in new_annotations.values():
            json.dump(annotation, file)
            file.write('\n')

# Specify the paths to the directories
directories = ['train', 'test', 'valid']

for directory in directories:
    # Construct the paths to the input and output annotation files
    coco_annotation_path = f'C:/Users/chris/foosball-dataset/foosball_coco/{directory}/metadata_coco_{directory}.json'
    output_annotation_path = f'C:/Users/chris/foosball-dataset/foosball_coco/{directory}/metadata.jsonl'

    # Check if the annotation file exists in the directory
    if os.path.exists(coco_annotation_path):
        # Perform the conversion
        convert_annotations(coco_annotation_path, output_annotation_path)
