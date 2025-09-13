import cv2
import numpy as np
import os
import albumentations as A
from tqdm import tqdm # For a progress bar

# --- Configuration ---
# Base directories
INPUT_BASE_DIR = '/home/freystec/foosball_data/new_puppets_dataset' # Your current data folder
OUTPUT_BASE_DIR = '/home/freystec/foosball_data/new_puppets_dataset_cropped' # Where to save the processed data

# Subdirectories for train/val
SUBDIRS = ['train', 'val']

# Define image type rules
TOPVIEW_PREFIXES = ("testvideo", "frame_", "2023-03-", "0000_all_labels_example")

# --- Helper Functions (from previous response, slightly adapted) ---
def get_image_type(filename):
    """Determines if an image is 'Topview' or 'Non-Topview' based on its filename."""
    for prefix in TOPVIEW_PREFIXES:
        if filename.startswith(prefix):
            return "Topview"
    return "Non-Topview"

def load_yolo_labels(label_path):
    """Loads YOLO format labels and converts them to [x_min, y_min, x_max, y_max, class_id] (normalized)."""
    boxes = []
    if os.path.exists(label_path):
        with open(label_path, 'r') as f:
            for line in f:
                parts = list(map(float, line.strip().split()))
                class_id = int(parts[0])
                # YOLO format: x_center, y_center, width, height (normalized)
                x_center, y_center, w, h = parts[1:]

                # Convert to x_min, y_min, x_max, y_max (normalized)
                x_min = x_center - w / 2
                y_min = y_center - h / 2
                x_max = x_center + w / 2
                y_max = y_center + h / 2

                boxes.append([x_min, y_min, x_max, y_max, class_id])
    return boxes

def save_yolo_labels(output_label_path, boxes):
    """Saves bounding boxes in YOLO format."""
    with open(output_label_path, 'w') as f:
        for box in boxes:
            # Albumentations output: x_min_norm, y_min_norm, x_max_norm, y_max_norm, class_id
            x_min_norm, y_min_norm, x_max_norm, y_max_norm, class_id = box
            
            # Convert back to YOLO format (x_center, y_center, width, height) normalized
            x_center_norm = (x_min_norm + x_max_norm) / 2
            y_center_norm = (y_min_norm + y_max_norm) / 2
            width_norm = x_max_norm - x_min_norm
            height_norm = y_max_norm - y_min_norm
            
            # Ensure values are within [0, 1] range after crop, before saving
            # Albumentations should generally handle this, but clipping adds robustness
            x_center_norm = np.clip(x_center_norm, 0.0, 1.0)
            y_center_norm = np.clip(y_center_norm, 0.0, 1.0)
            width_norm = np.clip(width_norm, 0.0, 1.0)
            height_norm = np.clip(height_norm, 0.0, 1.0)
            
            # Filter out boxes that became too small or invalid due to cropping
            # A common heuristic is to remove boxes with tiny width/height or those outside
            # Albumentations has remove_empty=True in BboxParams, but this additional check helps.
            if width_norm > 0.001 and height_norm > 0.001 and \
               x_center_norm > 0 and x_center_norm < 1 and \
               y_center_norm > 0 and y_center_norm < 1: 
                f.write(f"{int(class_id)} {x_center_norm:.6f} {y_center_norm:.6f} {width_norm:.6f} {height_norm:.6f}\n")

# --- Main Processing Logic ---
def process_dataset():
    """
    Processes images and labels based on their type, applying specific cropping rules.
    """
    for subdir in SUBDIRS:
        input_images_dir = os.path.join(INPUT_BASE_DIR, 'images', subdir)
        input_labels_dir = os.path.join(INPUT_BASE_DIR, 'labels', subdir)
        output_images_dir = os.path.join(OUTPUT_BASE_DIR, 'images', subdir)
        output_labels_dir = os.path.join(OUTPUT_BASE_DIR, 'labels', subdir)

        # Create output directories
        os.makedirs(output_images_dir, exist_ok=True)
        os.makedirs(output_labels_dir, exist_ok=True)

        print(f"Processing {subdir} directory...")
        image_files = [f for f in os.listdir(input_images_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        for img_filename in tqdm(image_files, desc=f"Cropping {subdir}"):
            image_path = os.path.join(input_images_dir, img_filename)
            base_name = os.path.splitext(img_filename)[0]
            label_name = base_name + '.txt'
            label_path = os.path.join(input_labels_dir, label_name)
            
            output_image_path = os.path.join(output_images_dir, img_filename)
            output_label_path = os.path.join(output_labels_dir, label_name)

            image = cv2.imread(image_path)
            if image is None:
                print(f"Skipping {img_filename}: Could not read image.")
                continue

            h_original, w_original = image.shape[:2]
            img_type = get_image_type(img_filename)

            x_min_crop = 0
            x_max_crop = w_original
            y_min_crop = 0 # Cropping only in x-direction, so y_min/y_max are always 0/height
            y_max_crop = h_original

            # --- Determine Cropping Parameters ---
            if img_type == "Non-Topview":
                if w_original == 1920:
                    x_min_crop = 320
                    x_max_crop = 1920 - 320 # 1600
                elif w_original < 1920 and w_original >= 500:
                    pixels_to_crop = int(w_original * 0.1667)
                    x_min_crop = pixels_to_crop
                    x_max_crop = w_original - pixels_to_crop
                elif w_original < 500:
                    # No cropping, x_min_crop and x_max_crop remain 0 and w_original
                    pass 
            elif img_type == "Topview":
                # Assuming Topview images are always 1920 wide as stated
                if w_original == 1920:
                    x_min_crop = 150
                    x_max_crop = 1920 - 150 # 1770
                else:
                    print(f"Warning: Topview image '{img_filename}' has width {w_original} != 1920. Applying no crop for safety.")
                    # Fallback to no crop if width is unexpected for Topview
                    pass 

            # Load labels in albumentations' format: [x_min, y_min, x_max, y_max, class_id] (normalized)
            bboxes = load_yolo_labels(label_path)
            
            # Prepare data for albumentations
            class_ids = [box[4] for box in bboxes]
            bboxes_for_aug = [box[:4] for box in bboxes] # just the coordinates x_min, y_min, x_max, y_max

            # Define the crop transform
            transform = A.Compose([
                A.Crop(x_min=x_min_crop, y_min=y_min_crop,
                       x_max=x_max_crop, y_max=y_max_crop),
            ], bbox_params=A.BboxParams(format='albumentations', label_fields=['class_ids'], 
                                        min_area=1, min_visibility=0.01, # Keep boxes that are at least 1 pixel area and 1% visible
                                        check_each_transform=True)) # 'remove_empty' removed
            # `check_each_transform=True` is useful if you have multiple transforms in a chain
            # `remove_empty=True` automatically removes boxes that become empty (width or height <= 0) after transformation

            try:
                augmented = transform(image=image, bboxes=bboxes_for_aug, class_ids=class_ids)
            except Exception as e:
                print(f"Error applying augmentation to {img_filename}: {e}")
                continue # Skip to next image

            cropped_image = augmented['image']
            adjusted_bboxes = augmented['bboxes']
            adjusted_class_ids = augmented['class_ids']

            # Combine adjusted bboxes with their class_ids
            final_bboxes = []
            for i, bbox in enumerate(adjusted_bboxes):
                final_bboxes.append(list(bbox) + [adjusted_class_ids[i]])
                    
            # Save the cropped image
            cv2.imwrite(output_image_path, cropped_image)

            # Save the adjusted labels
            save_yolo_labels(output_label_path, final_bboxes)
            # print(f"Processed {img_filename}: Cropped image saved to {output_image_path}, labels to {output_label_path}") # Suppress for tqdm
    print(f"Finished processing {subdir} directory.")


# --- Execute ---
if __name__ == "__main__":
    print(f"Starting dataset cropping from '{INPUT_BASE_DIR}' to '{OUTPUT_BASE_DIR}'...")
    process_dataset()
    print("Dataset cropping complete!")