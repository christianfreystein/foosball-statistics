import json
import os

ID_MAPPING = {
    1: 0,
    2: 1,
}
NEW_CATEGORIES = [
    {"id": 0, "name": "ball", "supercategory": "foosball balls"},
    {"id": 1, "name": "figure", "supercategory": "foosball figures"}
]
BASE_DIR = r"C:\Users\chris\foosball-dataset\big foosball dataset adjusted"
DATASET_TYPES = ['train', 'valid', 'test']


# DATASET_TYPES = ['train']


class COCOCategoryUpdater:
    def __init__(self, json_path, id_mapping, new_categories):
        self.json_path = json_path
        self.data = self.load_json()
        self.id_mapping = id_mapping
        self.new_categories = new_categories

    def load_json(self):
        with open(self.json_path, 'r') as file:
            return json.load(file)

    def save_json(self):
        with open(self.json_path, 'w') as file:
            json.dump(self.data, file, indent=4)

    def update_annotations(self):
        new_annotations = []
        for ann in self.data['annotations']:
            old_id = ann['category_id']
            if old_id in self.id_mapping:
                ann['category_id'] = self.id_mapping[old_id]
                new_annotations.append(ann)
        self.data['annotations'] = new_annotations

    def update_categories(self):
        self.data['categories'] = self.new_categories

    def perform_update(self):
        self.update_annotations()
        self.update_categories()
        self.save_json()


def update_all_categories(dataset_dirs, id_mapping, new_categories):
    for dataset_dir in dataset_dirs:
        json_path = os.path.join(dataset_dir, '_annotations.coco.json')
        updater = COCOCategoryUpdater(json_path, id_mapping, new_categories)
        updater.perform_update()


# Example usage:

dataset_dirs = [os.path.join(BASE_DIR, dataset_type) for dataset_type in DATASET_TYPES]

update_all_categories(dataset_dirs, ID_MAPPING, NEW_CATEGORIES)
