import json
import os
from datetime import datetime
import shutil
import random
from tqdm import tqdm


def update_and_merge_coco_datasets_in_one_file(dataset_dir):
    current_datetime = datetime.now().isoformat()

    new_data = {
        'info': {
            "year": current_datetime[:4],  # Extracting the year from the current date and time
            "version": "1",
            "description": "Taken from different resources from Roboflow and other websites",
            "contributor": "",
            "url": "no website",
            "date_created": current_datetime  # Using the current date and time
        },
        'licenses': [],
        'categories': [],
        'images': [],
        'annotations': []
    }

    img_path_dict = {}  # Dictionary to store image id to image path mapping
    image_id = 0  # New image id
    annotation_id = 0  # New annotation id

    # Iterate through each sub-dataset directory in the main dataset directory
    for subdir in tqdm(os.listdir(dataset_dir), desc="Processing datasets", unit="dataset"):
        subdir_path = os.path.join(dataset_dir, subdir)

        # Skip if not a directory
        if not os.path.isdir(subdir_path):
            continue

        # Iterate through each of train, valid, and test folders, if they exist
        for folder in ['train', 'valid', 'test']:
            folder_path = os.path.join(subdir_path, folder)

            # Skip if the folder does not exist
            if not os.path.exists(folder_path):
                continue

            # Iterate through each JSON file in the folder
            for json_filename in os.listdir(folder_path):
                if json_filename.endswith('.json'):
                    json_path = os.path.join(folder_path, json_filename)

                    # Load the COCO JSON file
                    with open(json_path, 'r') as file:
                        data = json.load(file)

                    # If this is the first file, use its categories
                    if not new_data['categories']:
                        new_data['categories'] = data['categories']
                        new_data['licenses'] = data['licenses']

                    image_id_mapping = {}  # Dictionary to store old image id to new image id mapping

                    # Updating the image ids and adding dataset source attribute
                    for img in data['images']:
                        old_image_id = img['id']  # Storing old image id
                        img['id'] = image_id
                        image_id_mapping[old_image_id] = image_id  # Storing the mapping of old image id to new image id

                        img['dataset_source'] = os.path.join(subdir, folder)  # Adding dataset source attribute
                        new_data['images'].append(img)
                        # Storing the image id and file name mapping
                        img_path_dict[image_id] = os.path.join(folder_path, img['file_name'])
                        image_id += 1

                    for ann in data['annotations']:
                        ann['image_id'] = image_id_mapping[ann['image_id']]  # Updating the image_id in annotations
                        ann['id'] = annotation_id
                        new_data['annotations'].append(ann)
                        annotation_id += 1

    # Saving the merged COCO JSON file
    # with open(output_json_path, 'w') as file:
    #    json.dump(new_data, file, indent=4)

    # Saving the image id to image path mapping as a JSON file
    # with open(output_img_path_dict, 'w') as file:
    #    json.dump(img_path_dict, file, indent=4)

    return new_data, img_path_dict


def split_data(all_data, ratios):
    num_images = len(all_data['images'])
    indices = list(range(num_images))
    random.shuffle(indices)

    split_points = [int(ratio * num_images) for ratio in ratios]
    split_indices = [indices[sum(split_points[:i]):sum(split_points[:i + 1])] for i in range(len(split_points))]

    split_data = []
    for indices in split_indices:
        data = {
            'info': all_data['info'],
            'licenses': all_data['licenses'],
            'categories': all_data['categories'],
            'images': [all_data['images'][i] for i in indices],
            'annotations': [ann for ann in all_data['annotations'] if ann['image_id'] in indices]
        }
        split_data.append(data)

    return split_data


def copy_images(indices, data, img_path_dict, target_dir):
    for i in tqdm(indices, desc="Copying images", unit="image"):
        src_img_path = img_path_dict[data['images'][i]['id']]
        dest_img_path = os.path.join(target_dir, os.path.basename(src_img_path))
        shutil.copy(src_img_path, dest_img_path)


def load_data(json_path):
    with open(json_path, 'r') as file:
        return json.load(file)


def save_data(data, output_path):
    with open(output_path, 'w') as file:
        json.dump(data, file, indent=4)

def ensure_dir_exists(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

def update_and_merge_coco_datasets(dataset_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    # Merge all data
    all_data, img_path_dict = update_and_merge_coco_datasets_in_one_file(dataset_dir)

    # Split data into train, valid, and test
    train_data, valid_data, test_data = split_data(all_data, [0.7, 0.2, 0.1])

    # Ensure the directories for train, valid, and test exist
    train_dir = os.path.join(output_dir, 'train')
    valid_dir = os.path.join(output_dir, 'valid')
    test_dir = os.path.join(output_dir, 'test')
    ensure_dir_exists(train_dir)
    ensure_dir_exists(valid_dir)
    ensure_dir_exists(test_dir)

    # Save new datasets
    save_data(train_data, os.path.join(output_dir, 'train', '_annotations.coco.json'))
    save_data(valid_data, os.path.join(output_dir, 'valid', '_annotations.coco.json'))
    save_data(test_data, os.path.join(output_dir, 'test', '_annotations.coco.json'))

    # Copy images to corresponding directories
    copy_images(range(len(train_data['images'])), train_data, img_path_dict, os.path.join(output_dir, 'train'))
    copy_images(range(len(valid_data['images'])), valid_data, img_path_dict, os.path.join(output_dir, 'valid'))
    copy_images(range(len(test_data['images'])), test_data, img_path_dict, os.path.join(output_dir, 'test'))


# Usage
dataset_dir = r"C:\Users\chris\foosball-dataset\foosball datasets"  # Replace with the path to the directory
# containing your dataset folders
output_dir = r"C:\Users\chris\foosball-dataset\merged_foosball_dataset"  # Replace with the desired output directory path
update_and_merge_coco_datasets(dataset_dir, output_dir)

print('finished')
