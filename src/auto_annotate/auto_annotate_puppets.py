import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from typing import Dict
import os
import shutil

# Assuming colors are imported from ultralytics.utils.plotting, as in your original function
from ultralytics.utils.plotting import colors

def visualize_image_annotations_and_save(image_path: str, txt_path: str, label_map: Dict[int, str], output_save_path: str, visualize_segmentation: bool = True, show_labels: bool = False):
    """
    Visualize YOLO annotations (bounding boxes and class labels) on an image and save the result.

    This function reads an image and its corresponding annotation file in YOLO format, then
    draws bounding boxes around detected objects and labels them with their respective class names.
    The bounding box colors are assigned based on the class ID, and the text color is dynamically
    adjusted for readability, depending on the background color's luminance.

    Args:
        image_path (str): The path to the image file to annotate.
        txt_path (str): The path to the annotation file in YOLO format.
        label_map (Dict[int, str]): A dictionary that maps class IDs (integers) to class labels (strings).
        output_save_path (str): The full path where the visualized image should be saved.
        visualize_segmentation (bool): If True, segmentation masks will be drawn. If False,
                                       only bounding boxes (either from the original annotation
                                       or derived from segmentation) will be drawn.
        show_labels (bool): If True, class labels will be displayed on the visualized image.
                            If False, labels will be hidden.
    """
    img = np.array(Image.open(image_path))
    img_height, img_width = img.shape[:2]
    annotations = [] # Stores (x, y, w, h, class_id, segment_points)

    # Check if the annotation file exists. If not, handle it gracefully.
    if not os.path.exists(txt_path):
        print(f"Warning: Annotation file not found for {os.path.basename(image_path)} at {txt_path}. Saving original image.")
        # If no annotations, just save the original image
        plt.imsave(output_save_path, img)
        return

    with open(txt_path, encoding="utf-8") as file:
        for line in file:
            try:
                parts = list(map(float, line.split()))
                class_id = int(parts[0])
                
                if len(parts) == 5:
                    # Bounding box format: class_id x_c y_c w h
                    x_center, y_center, width, height = parts[1:]
                    x = (x_center - width / 2) * img_width
                    y = (y_center - height / 2) * img_height
                    w = width * img_width
                    h = height * img_height
                    annotations.append((x, y, w, h, class_id, None)) # None for segmentation
                else:
                    # Segmentation format: class_id x1 y1 x2 y2 ... xn yn
                    segment_points_normalized = parts[1:]
                    segment_points_pixels = np.array([(segment_points_normalized[i] * img_width, segment_points_normalized[i+1] * img_height)
                                                       for i in range(0, len(segment_points_normalized), 2)], dtype=np.int32)
                    
                    if len(segment_points_pixels) > 0:
                        # Calculate approximate bounding box from segment points for both visualization and new YOLO
                        x_min, y_min = np.min(segment_points_pixels, axis=0)
                        x_max, y_max = np.max(segment_points_pixels, axis=0)
                        x = x_min
                        y = y_min
                        w = x_max - x_min
                        h = y_max - y_min
                        annotations.append((x, y, w, h, class_id, segment_points_pixels))
                    else:
                        print(f"Warning: Empty segmentation for class {class_id} in {txt_path}")
                        continue
            except ValueError as e:
                print(f"Error parsing line in {txt_path}: '{line.strip()}'. Error: {e}")
                continue

    # Set up figure size to match image dimensions for exact pixel output
    # Matplotlib's figsize is in inches, dpi is dots per inch
    # So, width_in_inches = img_width / dpi, height_in_inches = img_height / dpi
    # To get exact pixel size, set dpi to 1 and figsize to (img_width, img_height) if using pixels directly,
    # or ensure figsize * dpi matches pixel dimensions.
    # A common dpi for screens is 100, so we can calculate figsize based on that.
    # Alternatively, you can explicitly set dpi to 1 and figsize to the pixel dimensions,
    # but be aware this can make other matplotlib elements (fonts, lines) appear tiny.
    # A simpler way to get the exact original image size without messing with dpi too much
    # is to use the `imsave` function directly on the annotated numpy array,
    # but this requires rendering the annotations into the numpy array first.
    # For now, let's aim for a high quality output that matches resolution as closely as possible.

    # Option 1: Use a fixed DPI (e.g., 100) and calculate figsize
    # dpi_val = 100
    # fig, ax = plt.subplots(1, figsize=(img_width / dpi_val, img_height / dpi_val), dpi=dpi_val)

    # Option 2: Set DPI to 1 and figsize to image dimensions (results in very fine lines/text if text is enabled)
    # This is often the best for exact pixel mapping for images with no axes/text.
    fig = plt.figure(frameon=False) # No frame around the plot
    fig.set_size_inches(img_width / fig.dpi, img_height / fig.dpi) # Set size in inches based on current dpi
    
    ax = plt.Axes(fig, [0., 0., 1., 1.]) # Full axes covering the figure
    ax.set_axis_off() # Turn off all axes decorations
    fig.add_axes(ax)

    ax.imshow(img) # Display the base image

    for x, y, w, h, label, segment_points in annotations:
        color_bgr = colors(label, True) # Get BGR color
        color_rgb = (color_bgr[2] / 255, color_bgr[1] / 255, color_bgr[0] / 255) # Convert to RGB for matplotlib

        if visualize_segmentation and segment_points is not None and len(segment_points) > 0:
            # Draw segmentation mask
            polygon = plt.Polygon(segment_points, closed=True, fill=True, color=color_rgb, alpha=0.4)
            ax.add_patch(polygon)
            # Draw polygon outline
            outline_polygon = plt.Polygon(segment_points, closed=True, fill=False, edgecolor=color_rgb, linewidth=2)
            ax.add_patch(outline_polygon)
        else:
            # Draw bounding box if no segment (or if we only have bbox data)
            rect = plt.Rectangle((x, y), w, h, linewidth=2, edgecolor=color_rgb, facecolor="none")
            ax.add_patch(rect)

        # Add text label only if show_labels is True
        if show_labels:
            label_text = label_map.get(label, f"Class {label}")
            luminance = 0.2126 * color_rgb[0] + 0.7152 * color_rgb[1] + 0.0722 * color_rgb[2]
            text_color = "white" if luminance < 0.5 else "black"
            
            # Adjust text position for better visibility if it goes off-screen
            text_y_pos = y - 5
            if text_y_pos < 0:
                text_y_pos = y + h + 5 # Place below if it's too high

            ax.text(x, text_y_pos, label_text, color=text_color, backgroundcolor=color_rgb,
                    fontsize=8, ha='left', va='bottom')

    # ax.axis('off') # Already handled by ax.set_axis_off() and fig setup
    # plt.tight_layout() # Not needed with frameon=False, Axes at [0.,0.,1.,1.] and bbox_inches='tight'
    plt.savefig(output_save_path, dpi=fig.dpi, bbox_inches='tight', pad_inches=0) # Save with explicit dpi and tight bounding box
    plt.close(fig) # Close the figure to free up memory

def save_bboxes_from_segmentations(txt_path: str, new_bbox_save_path: str, img_width: int, img_height: int):
    """
    Reads a YOLO annotation file (potentially with segmentation masks) and saves
    only the bounding box information (derived from segmentation if present,
    otherwise uses existing bounding boxes) to a new YOLO format file.

    Args:
        txt_path (str): The path to the original annotation file in YOLO format.
        new_bbox_save_path (str): The path where the new YOLO bounding box annotation file should be saved.
        img_width (int): The width of the corresponding image.
        img_height (int): The height of the corresponding image.
    """
    new_annotations = []

    if not os.path.exists(txt_path):
        print(f"Warning: Annotation file not found at {txt_path}. No new bounding boxes will be saved.")
        return

    with open(txt_path, encoding="utf-8") as file:
        for line in file:
            try:
                parts = list(map(float, line.split()))
                class_id = int(parts[0])
                
                if len(parts) == 5:
                    # Bounding box format: class_id x_c y_c w h - use as is
                    x_center, y_center, width, height = parts[1:]
                    new_annotations.append(f"{class_id} {x_center} {y_center} {width} {height}")
                else:
                    # Segmentation format: class_id x1 y1 x2 y2 ... xn yn
                    segment_points_normalized = parts[1:]
                    segment_points_pixels = np.array([(segment_points_normalized[i] * img_width, segment_points_normalized[i+1] * img_height)
                                                       for i in range(0, len(segment_points_normalized), 2)], dtype=np.int32)
                    
                    if len(segment_points_pixels) > 0:
                        x_min, y_min = np.min(segment_points_pixels, axis=0)
                        x_max, y_max = np.max(segment_points_pixels, axis=0)
                        
                        # Convert pixel coordinates to normalized YOLO format
                        x_center_norm = ((x_min + x_max) / 2) / img_width
                        y_center_norm = ((y_min + y_max) / 2) / img_height
                        width_norm = (x_max - x_min) / img_width
                        height_norm = (y_max - y_min) / img_height

                        new_annotations.append(f"{class_id} {x_center_norm:.6f} {y_center_norm:.6f} {width_norm:.6f} {height_norm:.6f}")
                    else:
                        print(f"Warning: Empty segmentation for class {class_id} in {txt_path}, skipping bbox extraction.")
            except ValueError as e:
                print(f"Error parsing line in {txt_path}: '{line.strip()}'. Error: {e}")
                continue

    with open(new_bbox_save_path, "w", encoding="utf-8") as f:
        for annotation_line in new_annotations:
            f.write(annotation_line + "\n")


# --- Integration with your auto_annotate workflow ---

# First, run your auto_annotate (as you already have it)
from ultralytics.data.annotator import auto_annotate

# Define paths
data_input_dir = "/home/freystec/foosball_data/complete_labels"
det_model_path = "foosball-statistics/weights/yolov8n_puppets_first_iteration.pt"
sam_model_path = "/home/freystec/foosball-statistics/weights/sam2.1_b.pt"
auto_annotate_output_dir = "/home/freystec/foosball_data/labels"
visualization_output_dir = "/home/freystec/foosball_data/images_visualized"
new_bboxes_output_dir = "/home/freystec/foosball_data/new_bboxes_from_segments" # New directory for refined bounding boxes

# Configuration for visualization and bbox extraction
VISUALIZE_SEGMENTATION = False # Set to True to visualize segmentation, False to visualize only bounding boxes
SHOW_LABELS = False # Set to True to display class labels, False to hide them
GENERATE_NEW_BBOXES = True # Set to True to generate new YOLO bounding box files from segmentations

# Ensure output directories exist
os.makedirs(visualization_output_dir, exist_ok=True)
os.makedirs(auto_annotate_output_dir, exist_ok=True) 
if GENERATE_NEW_BBOXES:
    os.makedirs(new_bboxes_output_dir, exist_ok=True)

print("Starting auto-annotation...")
auto_annotate(
    data=data_input_dir,
    det_model=det_model_path,
    sam_model=sam_model_path,
    output_dir=auto_annotate_output_dir
)
print("Auto-annotation complete.")

# Define your label map based on your detection model's classes
# IMPORTANT: Adjust this if your model detects other classes or if 'puppet' has a different ID
label_map = {
        1: "puppet",
        0: "ball",
    # Add other classes if your 'det_model' is trained to detect them
}

print(f"Starting processing of annotations in {auto_annotate_output_dir}...")

# Path to the images that auto_annotate processed (it usually creates a symlink or copy)
# This assumes the images within the auto_annotate_output_dir/images/train
# are the ones you want to visualize with the labels in auto_annotate_output_dir/labels/train
images_to_visualize_dir = os.path.join(auto_annotate_output_dir, "images", "train")
labels_dir = os.path.join(auto_annotate_output_dir) # Labels are usually directly in the output_dir

# Fallback if auto_annotate does not copy images to its output dir but just refers to original
if not os.path.exists(images_to_visualize_dir) or not os.listdir(images_to_visualize_dir):
    print(f"Warning: {images_to_visualize_dir} is empty or does not exist. Using original input image directory for processing.")
    images_to_visualize_dir = data_input_dir


# Loop through the images and visualize/extract new bounding boxes
for img_filename in os.listdir(images_to_visualize_dir):
    if img_filename.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):
        image_path = os.path.join(images_to_visualize_dir, img_filename)
        txt_filename = os.path.splitext(img_filename)[0] + '.txt'
        txt_path = os.path.join(labels_dir, txt_filename)
        
        # Open image once to get dimensions for normalization
        try:
            with Image.open(image_path) as img_pil:
                img_width, img_height = img_pil.size
        except Exception as e:
            print(f"Error opening image {image_path}: {e}. Skipping.")
            continue

        # Define the output path for the visualized image
        output_save_path = os.path.join(visualization_output_dir, img_filename)
        
        # Visualize the image with annotations
        visualize_image_annotations_and_save(image_path, txt_path, label_map, output_save_path, 
                                            visualize_segmentation=VISUALIZE_SEGMENTATION,
                                            show_labels=SHOW_LABELS) # Pass the new parameter
        print(f"Visualized {img_filename} and saved to {output_save_path}")

        # Generate and save new bounding box files if enabled
        if GENERATE_NEW_BBOXES:
            new_bbox_txt_path = os.path.join(new_bboxes_output_dir, txt_filename)
            save_bboxes_from_segmentations(txt_path, new_bbox_txt_path, img_width, img_height)
            print(f"Generated new bounding box file for {img_filename} at {new_bbox_txt_path}")

print("All processing complete.")