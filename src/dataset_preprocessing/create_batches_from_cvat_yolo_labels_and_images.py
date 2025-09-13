import os
import shutil
import random

# --- Configuration ---
source_images_folder = "/home/freystec/foosball_data/complete_labels"
source_labels_folder = "/home/freystec/foosball_data/complete_labels_cvat_yolo/obj_train_data"
base_output_dir = "/home/freystec/foosball_data" # Parent directory for all new batch folders

num_batches = 4
yolo_batch_folder_prefix = "complete_labels_cvat_yolo_batch_"
images_batch_folder_prefix = "complete_labels_images_batch_"

# Dataset-level files content (will be copied into each batch's YOLO folder)
# Note: 'train' path is relative to the directory where obj.data is located (e.g., complete_labels_cvat_yolo_batch_0)
obj_data_template = """classes = {num_classes}
train = obj_train_data/train.txt
names = obj.names
backup = backup/
"""
obj_names_content = """
ball (slow, easy)
ball (slow, moderate)
ball (slow, hard)
ball (moderately fast)
ball (very fast)
ball (unclear)
ball (slow, impossible)
buffer
adjusting ring
playable area
goal counter
table key points
bar
goal key points
puppet
unclear
"""
class_names_count = len(obj_names_content.strip().split('\n')) # Calculate from content

# --- Ensure source folders exist ---
if not os.path.exists(source_images_folder):
    print(f"Error: Source images folder '{source_images_folder}' not found.")
    print("Please ensure this path is correct and contains your .jpg image files.")
    exit()
if not os.path.exists(source_labels_folder):
    print(f"Error: Source labels folder '{source_labels_folder}' not found.")
    print("Please ensure this path is correct and contains your .txt label files.")
    exit()

# --- Get all image-label base names that have a complete pair ---
image_label_basenames = []
image_files_in_src = {os.path.splitext(f)[0] for f in os.listdir(source_images_folder) if f.endswith(".jpg")}
label_files_in_src = {os.path.splitext(f)[0] for f in os.listdir(source_labels_folder) if f.endswith(".txt")}

# Find common basenames (those that have both an image and a label)
valid_basenames = list(image_files_in_src.intersection(label_files_in_src))

if not valid_basenames:
    print(f"No matching image (.jpg) and label (.txt) pairs found between '{source_images_folder}' and '{source_labels_folder}'. Exiting.")
    exit()

random.shuffle(valid_basenames)
print(f"Found {len(valid_basenames)} complete image-label pairs to distribute.")

# --- Calculate split sizes ---
total_pairs = len(valid_basenames)
split_size = total_pairs // num_batches # Integer division
remainder = total_pairs % num_batches # For distributing remaining items evenly

# --- Create batch folders and distribute files ---
current_index = 0
for i in range(num_batches):
    # Define paths for the current batch
    current_yolo_batch_root = os.path.join(base_output_dir, f"{yolo_batch_folder_prefix}{i}")
    current_images_batch_root = os.path.join(base_output_dir, f"{images_batch_folder_prefix}{i}")

    # Create directories for the YOLO structure within the batch root
    # Images and labels will go directly into obj_train_data for the YOLO structure
    current_obj_train_data_path = os.path.join(current_yolo_batch_root, "obj_train_data")

    os.makedirs(current_obj_train_data_path, exist_ok=True)
    os.makedirs(current_images_batch_root, exist_ok=True) # Create the separate images folder

    print(f"\n--- Setting up Batch {i} ---")
    print(f"Created YOLO folder: {current_yolo_batch_root}")
    print(f"Created images folder: {current_images_batch_root}")

    # Determine batch items
    batch_end_index = current_index + split_size + (1 if i < remainder else 0)
    batch_items = valid_basenames[current_index:batch_end_index]

    copied_labels_count = 0
    copied_images_count = 0
    train_txt_lines = []

    for base_name in batch_items:
        original_img_path = os.path.join(source_images_folder, base_name + ".jpg")
        original_txt_path = os.path.join(source_labels_folder, base_name + ".txt")

        # Destination for YOLO labels (.txt)
        dest_txt_path = os.path.join(current_obj_train_data_path, base_name + ".txt")
        # Destination for images (.jpg) - inside obj_train_data for the YOLO structure
        dest_img_for_yolo_path = os.path.join(current_obj_train_data_path, base_name + ".jpg")
        # Destination for images (.jpg) - in the separate images batch folder
        dest_img_for_separate_folder_path = os.path.join(current_images_batch_root, base_name + ".jpg")

        # Copy label file
        shutil.copy2(original_txt_path, dest_txt_path)
        copied_labels_count += 1

        # Copy image file to BOTH locations
        shutil.copy2(original_img_path, dest_img_for_yolo_path) # For YOLO structure
        shutil.copy2(original_img_path, dest_img_for_separate_folder_path) # For separate image folder
        copied_images_count += 1
        
        # Path for train.txt (relative to the batch's obj.data)
        train_txt_lines.append(f"obj_train_data/{base_name}.jpg") 

    print(f"Copied {copied_labels_count} label files and {copied_images_count} image files to Batch {i}.")

    # Generate train.txt for this batch
    train_txt_path = os.path.join(current_obj_train_data_path, "train.txt")
    with open(train_txt_path, "w") as f:
        for line in train_txt_lines:
            f.write(line + "\n")
    print(f"Generated train.txt for Batch {i} with {len(train_txt_lines)} entries.")

    # Create obj.data for this batch
    obj_data_path = os.path.join(current_yolo_batch_root, "obj.data")
    with open(obj_data_path, "w") as f:
        f.write(obj_data_template.format(num_classes=class_names_count))
    print(f"Created obj.data for Batch {i}.")

    # Create obj.names for this batch
    obj_names_path = os.path.join(current_yolo_batch_root, "obj.names")
    with open(obj_names_path, "w") as f:
        f.write(obj_names_content)
    print(f"Created obj.names for Batch {i}.")

    current_index = batch_end_index

print("\nAll batches generated successfully!")
print(f"Check '{base_output_dir}' for your new batch folders.")