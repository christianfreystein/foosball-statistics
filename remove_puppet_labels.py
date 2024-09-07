import os

# Define the source and destination directories
source_folder = r"D:\labels\big_foosball_dataset_val\obj_train_data_old"
destination_folder = r"D:\labels\big_foosball_dataset_val\obj_train_data"

# Create the destination folder if it doesn't exist
if not os.path.exists(destination_folder):
    os.makedirs(destination_folder)

# Loop through all the files in the source folder
for filename in os.listdir(source_folder):
    if filename.endswith('.txt'):
        # Construct the full file paths
        source_file = os.path.join(source_folder, filename)
        destination_file = os.path.join(destination_folder, filename)

        # Open the source file and read its lines
        with open(source_file, 'r') as file:
            lines = file.readlines()

        # Filter out the lines that don't start with '0'
        filtered_lines = [line for line in lines if line.startswith('0')]

        # Write the filtered lines to the destination file
        with open(destination_file, 'w') as file:
            file.writelines(filtered_lines)

print("Processing complete.")
