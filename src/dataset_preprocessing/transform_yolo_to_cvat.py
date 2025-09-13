import os
import shutil

# --- Configuration ---
source_label_folder = "/home/freystec/foosball_data/new_bboxes_from_segments"
destination_base_folder = "/home/freystec/foosball_data/complete_labels_cvat_yolo"
obj_train_data_subfolder = "obj_train_data"
output_train_txt_name = "train.txt"
output_obj_data_name = "obj.data"
output_obj_names_name = "obj.names"

# Your class names (from your previous prompt)
class_names = [
    "ball (slow, easy)",
    "ball (slow, moderate)",
    "ball (slow, hard)",
    "ball (moderately fast)",
    "ball (very fast)",
    "ball (unclear)",
    "ball (slow, impossible)",
    "buffer",
    "adjusting ring",
    "playable area",
    "counter",
    "table key points",
    "bar",
    "goal key points",
    "puppet",
    "unclear"
]

# --- Label Remapping Configuration ---
# This dictionary maps old class IDs to new class IDs.
# For example: {14: 15} means change class ID 14 to 15.
# IMPORTANT: Class IDs are 0-indexed in YOLO label files.
# So if "puppet" is the 15th item in your class_names list (index 14),
# and you want to change its ID to 15, then the mapping would be {14: 15}.
label_remap = {1: 14} # Example: Remap puppet (original index 14) to new ID 15


# --- Create Destination Folders ---
full_destination_obj_train_data_path = os.path.join(destination_base_folder, obj_train_data_subfolder)

os.makedirs(full_destination_obj_train_data_path, exist_ok=True)
print(f"Created directory: {full_destination_obj_train_data_path}")

# --- Copy and Process Label Files ---
copied_and_processed_files_count = 0
for filename in os.listdir(source_label_folder):
    if filename.endswith(".txt"):
        source_path = os.path.join(source_label_folder, filename)
        destination_path = os.path.join(full_destination_obj_train_data_path, filename)

        # Read content, remap, and write
        with open(source_path, 'r') as f_in:
            lines = f_in.readlines()

        modified_lines = []
        for line in lines:
            parts = line.strip().split()
            if parts:
                class_id = int(parts[0])
                if class_id in label_remap:
                    new_class_id = label_remap[class_id]
                    parts[0] = str(new_class_id)
                    print(f"  Remapped class ID {class_id} to {new_class_id} in {filename}")
                modified_lines.append(" ".join(parts) + "\n")
            else:
                modified_lines.append(line) # Keep empty lines if any

        with open(destination_path, 'w') as f_out:
            f_out.writelines(modified_lines)
            
        copied_and_processed_files_count += 1

print(f"Copied and processed {copied_and_processed_files_count} label files from '{source_label_folder}' to '{full_destination_obj_train_data_path}'.")

# --- Generate train.txt ---
train_txt_path = os.path.join(destination_base_folder, output_train_txt_name)
image_paths = []
for filename in os.listdir(full_destination_obj_train_data_path):
    if filename.endswith(".txt"):
        # Replace .txt with .jpg to get the image name relative to the 'data' prefix
        image_name = filename.replace(".txt", ".jpg")
        image_paths.append(f"data/{obj_train_data_subfolder}/{image_name}") # Note: 'data/' is a convention for obj.data

with open(train_txt_path, "w") as f:
    for path in image_paths:
        f.write(path + "\n")
print(f"Generated {train_txt_path} with {len(image_paths)} image paths.")

# --- Create obj.data ---
obj_data_path = os.path.join(destination_base_folder, output_obj_data_name)
num_classes = len(class_names)
obj_data_content = f"""classes = {num_classes}
train = data/{output_train_txt_name}
names = data/{output_obj_names_name}
backup = backup/
"""
with open(obj_data_path, "w") as f:
    f.write(obj_data_content)
print(f"Created {obj_data_path}.")

# --- Create obj.names ---
obj_names_path = os.path.join(destination_base_folder, output_obj_names_name)
with open(obj_names_path, "w") as f:
    for name in class_names:
        f.write(name + "\n")
print(f"Created {obj_names_path} with {num_classes} class names.")

print("\nConversion complete! Your CVAT YOLO labels are now prepared.")
print(f"The main directory is: {destination_base_folder}")