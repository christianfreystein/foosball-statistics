import os

# Define the folder containing the .txt files
folder_path = r"D:\labels\bratzdrum_labels_train\obj_train_data"

# Define the output file path
output_file_path = 'output.txt'

# Define the prefix for the image paths
prefix = 'data/obj_train_data/'

# Get a list of all .txt files in the folder
txt_files = [f for f in os.listdir(folder_path) if f.endswith('.txt')]

# Open the output file in write mode
with open(output_file_path, 'w') as output_file:
    # Iterate over each .txt file in the folder
    for txt_file in txt_files:
        # Replace the .txt extension with .jpg and add the prefix
        image_path = prefix + txt_file.replace('.txt', '.jpg')
        # Write the image path to the output file
        output_file.write(image_path + '\n')

print(f"Output written to {output_file_path}")

