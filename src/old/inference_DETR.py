import os
import torch
import glob
from transformers import AutoImageProcessor, AutoModelForObjectDetection
from PIL import Image, ImageDraw
from tqdm import tqdm
import shutil  # Import shutil to copy files
import json


def load_model(checkpoint):
    image_processor = AutoImageProcessor.from_pretrained(checkpoint)
    model = AutoModelForObjectDetection.from_pretrained(checkpoint)
    return image_processor, model


def detect_objects(image, image_processor, model):
    with torch.no_grad():
        inputs = image_processor(images=image, return_tensors="pt")
        outputs = model(**inputs)
        target_sizes = torch.tensor([image.size[::-1]])
        results = image_processor.post_process_object_detection(outputs, threshold=0.5, target_sizes=target_sizes)[0]
    return results


def draw_boxes(image, results, model):
    draw = ImageDraw.Draw(image)
    for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
        box = [round(i, 2) for i in box.tolist()]
        x, y, x2, y2 = tuple(box)
        draw.rectangle((x, y, x2, y2), outline="red", width=1)
        draw.text((x, y), model.config.id2label[label.item()], fill="white")
    return image


def process_images(input_folder, output_folder, checkpoint):
    # Check if the output folder exists, create it if it does not
    os.makedirs(output_folder, exist_ok=True)

    # Copy the config.json and preprocessor_config.json from checkpoint path to output folder
    shutil.copy(os.path.join(checkpoint, 'config.json'), output_folder)
    shutil.copy(os.path.join(checkpoint, 'preprocessor_config.json'), output_folder)

    image_processor, model = load_model(checkpoint)
    image_paths = glob.glob(f"{input_folder}/*.jpg")

    # Creating a dictionary to store results and other information
    output_data = {
        'checkpoint': checkpoint,
        'input_folder': input_folder,
        'output_folder': output_folder,
        'results': []
    }

    for image_path in tqdm(image_paths, desc="Processing images", unit="image"):
        image = Image.open(image_path)
        results = detect_objects(image, image_processor, model)
        image_with_boxes = draw_boxes(image, results, model)
        output_path = os.path.join(output_folder, os.path.basename(image_path))

        # Saving image with boxes
        image_with_boxes.save(output_path)

        # Preparing the results for JSON serialization
        results_for_json = {
            'image_path': os.path.basename(image_path),
            'scores': results['scores'].tolist(),
            'labels': results['labels'].tolist(),
            'boxes': results['boxes'].tolist()
        }
        output_data['results'].append(results_for_json)

    # Writing results to a JSON file in the output folder
    json_output_path = os.path.join(output_folder, 'results.json')
    with open(json_output_path, 'w') as json_file:
        json.dump(output_data, json_file, indent=4)

# Example usage
checkpoint = r"C:\Users\chris\Foosball Detector\detr-resnet-50_finetuned_merged_foosball_dataset"
input_folder = r"C:\Users\chris\foosball-dataset\visualization test results\test images"
output_folder = r"C:\Users\chris\foosball-dataset\visualization test results\results test images 2"

process_images(input_folder, output_folder, checkpoint)

