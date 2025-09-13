import os
import cv2
import numpy as np # Needed for array operations

# --- Configuration ---
CROPPED_DATA_BASE_DIR = '/home/freystec/Foosball_Datasets/test_data_cropped' # Your cropped data folder
SUBDIRS = ['train', 'val']

# Define your class names here. 
# IMPORTANT: Adjust this `CLASS_NAMES` list to match ALL class IDs in your YOLO labels!
CLASS_NAMES = {
    0: 'ball', 
    # Add other class IDs if they exist in your label files, e.g.:
    # 1: 'player',
    # 2: 'table_edge',
}

# --- Helper Function to load YOLO labels and convert to absolute pixel coordinates ---
def load_yolo_labels_abs(label_path, img_width, img_height):
    """
    Loads YOLO format labels from a .txt file and converts them to
    [class_id, x_min_abs, y_min_abs, x_max_abs, y_max_abs] format.
    """
    boxes = []
    if os.path.exists(label_path):
        with open(label_path, 'r') as f:
            for line in f:
                try:
                    parts = list(map(float, line.strip().split()))
                    class_id = int(parts[0])
                    # YOLO format: x_center, y_center, width, height (normalized)
                    x_center_norm, y_center_norm, w_norm, h_norm = parts[1:]

                    # Convert to absolute pixel coordinates (x_min, y_min, x_max, y_max)
                    x_center_abs = int(x_center_norm * img_width)
                    y_center_abs = int(y_center_norm * img_height)
                    w_abs = int(w_norm * img_width)
                    h_abs = int(h_norm * img_height)

                    x_min_abs = int(x_center_abs - w_abs / 2)
                    y_min_abs = int(y_center_abs - h_abs / 2)
                    x_max_abs = int(x_center_abs + w_abs / 2)
                    y_max_abs = int(y_center_abs + h_abs / 2)

                    boxes.append([class_id, x_min_abs, y_min_abs, x_max_abs, y_max_abs])
                except (ValueError, IndexError) as e:
                    print(f"Error parsing line in {label_path}: '{line.strip()}' - {e}")
    return boxes

# --- Main Visualization Function (Manual Drawing and Saving) ---
def visualize_and_save_cropped_dataset(num_samples_to_process=None):
    """
    Manually visualizes bounding box annotations on cropped images using OpenCV
    and saves the annotated images to a new directory.
    """
    for subdir in SUBDIRS:
        images_dir = os.path.join(CROPPED_DATA_BASE_DIR, 'images', subdir)
        labels_dir = os.path.join(CROPPED_DATA_BASE_DIR, 'labels', subdir)
        output_vis_dir = os.path.join(CROPPED_DATA_BASE_DIR, 'visualizations', subdir)
        
        # Create output directory if it doesn't exist
        os.makedirs(output_vis_dir, exist_ok=True)
        
        print(f"\n--- Processing and saving {subdir} subset (up to {num_samples_to_process if num_samples_to_process is not None else 'all'} samples) ---")
        
        image_files = [f for f in os.listdir(images_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        count = 0
        for img_filename in image_files:
            if num_samples_to_process is not None and count >= num_samples_to_process:
                break

            image_path = os.path.join(images_dir, img_filename)
            base_name = os.path.splitext(img_filename)[0]
            label_name = base_name + '.txt'
            label_path = os.path.join(labels_dir, label_name)

            if not os.path.exists(image_path):
                print(f"Skipping {img_filename}: Image file {image_path} not found.")
                continue

            # It's okay if a label file doesn't exist, we'll just draw the image without boxes
            if not os.path.exists(label_path):
                print(f"Warning: Label file {label_path} not found for {img_filename}. Image will be saved without annotations.")
            
            print(f"Processing: {img_filename}")
            
            # Load the image using OpenCV
            image = cv2.imread(image_path)
            if image is None:
                print(f"Error: Could not read image {image_path}. It might be corrupted or unsupported format.")
                continue
            
            img_height, img_width = image.shape[:2]

            # Load and convert labels to absolute pixel coordinates
            bboxes_abs = load_yolo_labels_abs(label_path, img_width, img_height)

            # Draw bounding boxes and labels
            for bbox in bboxes_abs:
                class_id, x_min, y_min, x_max, y_max = bbox
                
                # Ensure coordinates are within image bounds
                x_min = max(0, x_min)
                y_min = max(0, y_min)
                x_max = min(img_width - 1, x_max)
                y_max = min(img_height - 1, y_max)

                # Get class name and color
                class_name = CLASS_NAMES.get(class_id, f"Class {class_id}")
                color = (0, 255, 0) # Green BGR for bounding box
                text_color = (255, 255, 255) # White text
                
                # Draw rectangle
                cv2.rectangle(image, (x_min, y_min), (x_max, y_max), color, 2)

                # Draw label background
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.7
                font_thickness = 2
                (text_width, text_height), baseline = cv2.getTextSize(class_name, font, font_scale, font_thickness)
                
                text_y = y_min - 10 if y_min - 10 > text_height else y_min + text_height + 5
                text_x = x_min

                cv2.rectangle(image, (text_x, text_y - text_height - baseline), (text_x + text_width, text_y + baseline), color, -1)
                cv2.putText(image, class_name, (text_x, text_y), font, font_scale, text_color, font_thickness, cv2.LINE_AA)

            # Save the annotated image
            output_image_path = os.path.join(output_vis_dir, f"annotated_{img_filename}")
            cv2.imwrite(output_image_path, image)
            print(f"Saved annotated image to {output_image_path}")

            count += 1

# --- Execute Visualization and Saving ---
if __name__ == "__main__":
    # Call the manual visualization and saving function
    # Set num_samples_to_process=None to process ALL images,
    # or set it to an integer (e.g., 10) to process only a few.
    visualize_and_save_cropped_dataset(num_samples_to_process=10)