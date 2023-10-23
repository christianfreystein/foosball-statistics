import json
import os

# Base paths
base_dataset_path = r"C:\Users\chris\foosball-dataset\foosball_coco_3"
base_output_path = r"C:\Users\chris\foosball-dataset\foosball_coco_3"

def process_folder(folder_name, base_dataset_path, base_output_path):
    # Load the dataset
    input_file = os.path.join(base_dataset_path, folder_name, '_annotations.coco.json')
    with open(input_file, 'r') as file:
        dataset = json.load(file)

    # Folder containing the image files
    image_folder = os.path.join(base_dataset_path, folder_name)

    # Find the category id for "figures"
    figure_category_id = None
    for category in dataset['categories']:
        if category['name'] == 'figure':
            figure_category_id = category['id']
            break

    if figure_category_id is None:
        raise Exception('Category "figure" not found in dataset.')

    # Find image ids with annotations for the class "figures"
    figure_image_ids = set()
    for ann in dataset['annotations']:
        if ann['category_id'] == figure_category_id:
            figure_image_ids.add(ann['image_id'])

    # Filter images
    new_images = [img for img in dataset['images'] if img['id'] in figure_image_ids]

    # Delete images from folder
    for img in dataset['images']:
        if img['id'] not in figure_image_ids:
            img_path = os.path.join(image_folder, img['file_name'])
            if os.path.exists(img_path):
                os.remove(img_path)

    # Filter annotations
    new_annotations = [ann for ann in dataset['annotations'] if ann['image_id'] in figure_image_ids]

    # Update the dataset
    dataset['images'] = new_images
    dataset['annotations'] = new_annotations

    # Save the new dataset
    output_file = os.path.join(base_output_path, folder_name, '_annotations.coco.json')
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w') as file:
        json.dump(dataset, file)

# Process each folder separately
process_folder('test', base_dataset_path, base_output_path)
process_folder('train', base_dataset_path, base_output_path)
process_folder('valid', base_dataset_path, base_output_path)
