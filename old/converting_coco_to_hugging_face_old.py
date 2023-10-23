import json

# Load the COCO annotations file
with open(r"C:\Users\chris\foosball-dataset\merged_foosball_dataset\train\_annotations.coco.json", 'r') as f:
    coco_annotations = json.load(f)

# Initialize a counter for image_ids and object_ids
image_id = 0
object_id = 0

# Initialize the list to store converted annotations
huggingface_annotations = []

# Dictionary to map COCO category ids to names
category_mapping = {category['id']: category['name'] for category in coco_annotations['categories']}

# Loop over the images in the COCO annotations
for image in coco_annotations['images']:
    # Extract the relevant information from the COCO annotation
    file_name = image['file_name']
    width = image['width']
    height = image['height']

    # Collect the objects for this image
    objects = {'id': [], 'area': [], 'bbox': [], 'category': []}
    for annotation in coco_annotations['annotations']:
        if annotation['image_id'] == image['id']:
            bbox = annotation['bbox']
            area = annotation['area']
            category = annotation['category_id']
            object_id = annotation['id']
            # category = category_mapping[annotation['category_id']]
            objects['id'].append(object_id)
            objects['area'].append(area)
            objects['bbox'].append(bbox)
            objects['category'].append(category)

            # Increment the object_id
            # object_id += 1

    # Construct the annotation in Hugging Face format
    hf_annotation = {
        'image_id': image['id'],
        'file_name': file_name,
        'width': width,
        'height': height,
        'objects': objects
    }

    # Append the annotation to the list
    huggingface_annotations.append(hf_annotation)

    # Increment the image_id
    image_id += 1

# Save the converted annotations to a new file in JSONL format
with open(r"C:\Users\chris\foosball-dataset\merged_foosball_dataset\train\metadata.jsonl", 'w') as f:
    for annotation in huggingface_annotations:
        json.dump(annotation, f)
        f.write('\n')
