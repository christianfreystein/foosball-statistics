import os

# Define paths
root_folder = r"D:\New_Big_Foosball_Dataset\Big_Dataset_Foosball\labels" # Replace with the root folder path
output_folder = r"D:\New_Big_Foosball_Dataset\Big_Dataset_Foosball\processed_labels"  # Folder where processed labels will be saved

# Ensure output directory exists
if not os.path.exists(output_folder):
    os.makedirs(output_folder)


# Function to process the label files
def process_label_file(file_path, output_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()

    new_lines = []
    for line in lines:
        # Split the line into components
        parts = line.split()
        class_id = int(parts[0])

        # Check class ID and modify or discard lines
        if class_id in [0, 1, 2]:
            parts[0] = '0'
            new_lines.append(' '.join(parts))
        if class_id in [3, 4]:
            parts[0] = '1'
            new_lines.append(' '.join(parts))

    # Only save files that are not empty
    if new_lines:
        with open(output_path, 'w') as out_file:
            out_file.write('\n'.join(new_lines))


# Traverse the folder structure and process files
for subdir, dirs, files in os.walk(root_folder):
    if 'obj_train_data' in subdir:
        for file in files:
            if file.endswith('.txt'):
                file_path = os.path.join(subdir, file)
                output_path = os.path.join(output_folder, file)
                process_label_file(file_path, output_path)

print("Processing complete.")

import os




def map_txt_to_jpg(txt_folder, images_folder):
    # Step 1: Get all .txt files (strip the extensions)
    txt_files = {os.path.splitext(f)[0] for f in os.listdir(txt_folder) if f.endswith('.txt')}

    # Step 2: Traverse the images folder and find corresponding .jpg files
    mapping = {}
    matched_txt_files = set()  # To keep track of which txt files have corresponding jpg files

    for root, _, files in os.walk(images_folder):
        for file in files:
            if file.endswith('.jpg'):
                # Remove the extension from the image file name
                name = os.path.splitext(file)[0]
                if name in txt_files:
                    # If the name matches a .txt file, map it to the relative path of the image
                    relative_path = os.path.join(os.path.basename(root), file)
                    mapping[name] = relative_path
                    matched_txt_files.add(name)

    # Step 3: Find the txt files that don't have a matching .jpg
    unmatched_txt_files = txt_files - matched_txt_files

    return mapping, unmatched_txt_files


# Example usage:
txt_folder = r"D:\New_Big_Foosball_Dataset\Big_Dataset_Foosball\processed_labels"
images_folder = r"D:\New_Big_Foosball_Dataset\Big_Dataset_Foosball\images"

# Get the mapping and the unmatched .txt files
result, unmatched_txt_files = map_txt_to_jpg(txt_folder, images_folder)

# Output the mapping
print("Mapped files:")
for txt_name, image_path in result.items():
    print(f"{txt_name} => {image_path}")

# Output the unmatched .txt files
if unmatched_txt_files:
    print("\n.txt files without corresponding .jpg:")
    for txt_name in unmatched_txt_files:
        print(txt_name)
else:
    print("\nAll .txt files have corresponding .jpg files.")
# #
import os
import random
import shutil


def split_data(txt_folder, images_folder, mapping, split_ratio=0.95):
    # Step 1: Get all .txt file names (without extensions)
    txt_files = list(mapping.keys())

    # Step 2: Shuffle and split the .txt files (95% train, 5% val)
    random.shuffle(txt_files)
    split_index = int(len(txt_files) * split_ratio)
    train_files = txt_files[:split_index]
    val_files = txt_files[split_index:]

    # Step 3: Create the required folder structure
    labels_train_dir = os.path.join(txt_folder, 'labels', 'train')
    labels_val_dir = os.path.join(txt_folder, 'labels', 'val')
    images_train_dir = os.path.join(images_folder, 'images', 'train')
    images_val_dir = os.path.join(images_folder, 'images', 'val')

    os.makedirs(labels_train_dir, exist_ok=True)
    os.makedirs(labels_val_dir, exist_ok=True)
    os.makedirs(images_train_dir, exist_ok=True)
    os.makedirs(images_val_dir, exist_ok=True)

    # Step 4: Move the files to their respective folders
    for file_list, label_dir, image_dir in [(train_files, labels_train_dir, images_train_dir),
                                            (val_files, labels_val_dir, images_val_dir)]:
        for txt_name in file_list:
            # Move .txt files
            txt_src_path = os.path.join(txt_folder, f'{txt_name}.txt')
            txt_dest_path = os.path.join(label_dir, f'{txt_name}.txt')
            if os.path.exists(txt_src_path):
                shutil.copy(txt_src_path, txt_dest_path)

            # Move corresponding .jpg files
            if txt_name in mapping:
                jpg_src_path = os.path.join(images_folder, mapping[txt_name])
                jpg_dest_path = os.path.join(image_dir, os.path.basename(mapping[txt_name]))
                if os.path.exists(jpg_src_path):
                    shutil.copy(jpg_src_path, jpg_dest_path)

    print(f"Train set: {len(train_files)} files")
    print(f"Validation set: {len(val_files)} files")
    print("Data split completed!")


# Assuming `mapping` is obtained from the previous function you have
mapping = result

# Call the function to split and move the data
split_data(txt_folder, images_folder, mapping)

