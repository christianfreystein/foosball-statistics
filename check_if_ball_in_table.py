import json
from tqdm import tqdm

def read_json(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data


def write_json(data, file_path):
    with open(file_path, 'w') as file:
        json.dump(data, file, indent=4)


def is_point_in_area(px, py, vertices):
    num_vertices = len(vertices)
    inside = False
    x, y = px, py

    for i in range(num_vertices):
        j = (i + 1) % num_vertices
        xi, yi = vertices[i]
        xj, yj = vertices[j]

        if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi) + xi):
            inside = not inside

    return inside


def find_y_area(y):
    y_areas = {
        (61, 162): "Left 2",
        (162, 209): "Right 0",
        (209, 269): "Left 1",
        (269, 333): "Right 1",
        (333, 415): "Left 0",
        (415, 570): "Right 2"
    }

    for (y_min, y_max), area_name in y_areas.items():
        if y_min <= y < y_max:
            return area_name

    return "Outside"


def add_ball_status(data, vertices, image_width, image_height):
    for frame in tqdm(data, desc="Processing frames"):
        if 'current_ball_location' in frame:
            # Extract normalized coordinates
            normalized_x, normalized_y, _, _ = frame['current_ball_location']

            # Convert normalized coordinates to pixel values
            x_pixel = normalized_x * image_width
            y_pixel = normalized_y * image_height

            # Determine ball status based on y-coordinate
            ball_status = find_y_area(int(y_pixel))
            frame['ball_status'] = ball_status

    return data


# Define file paths
input_file_path = r"C:\Users\chris\Foosball Detector\First analyzed video\Haas_Klabunde_Moreland_Rue_Tracking_Video_long_all_data.json"
output_file_path = r"C:\Users\chris\Foosball Detector\First analyzed video\Haas_Klabunde_Moreland_Rue_Tracking_Video_long_all_data_with_ball_status.json"

# Image dimensions
image_width = 1280
image_height = 720

# Define the vertices of the quadrilateral
vertices = [
    (367, 560),  # Left down
    (923, 563),  # Right down
    (810, 112),  # Right up
    (471, 99)  # Left up
]

# Load the JSON data from the file
data = read_json(input_file_path)

# Add ball status to each frame
updated_data = add_ball_status(data, vertices, image_width, image_height)

# Save the updated data to a new file
write_json(updated_data, output_file_path)

print(f"JSON file updated successfully. The result is saved to {output_file_path}.")

# # Paths to the JSON files
# file2_path = r"C:\Users\chris\Foosball Detector\First analyzed video\Track_History_Haas_Klabunde_Moreland_Rue_Tracking_Video_long.json"
# new_json_path = r"C:\Users\chris\Foosball Detector\First analyzed video\Modified_Track_History_Haas_Klabunde_Moreland_Rue_Tracking_Video_long.json"
#
# # Read the JSON file
# data2 = read_json(file2_path)
#
# # Define the new entries to prepend
# new_entries = [[0, 0, 0, 0]] * 7
#
# # Prepend the new entries to the original list
# modified_list = new_entries + data2
#
# # Write the modified list to a new JSON file
# write_json(modified_list, new_json_path)
#
# print(f"Modified JSON saved to {new_json_path}")

# # Paths to the JSON files
# file1_path = r"C:\Users\chris\Foosball Detector\First analyzed video\Haas_Klabunde_Moreland_Rue_Tracking_Video_long.json"
# file2_path = r"C:\Users\chris\Foosball Detector\First analyzed video\Modified_Track_History_Haas_Klabunde_Moreland_Rue_Tracking_Video_long.json"
#
# # Read the JSON files
# data1 = read_json(file1_path)
# data2 = read_json(file2_path)


# import json
#
# # Define file paths
# file2_path = r"C:\Users\chris\Foosball Detector\First analyzed video\Haas_Klabunde_Moreland_Rue_Tracking_Video_long.json"
# file1_path = r"C:\Users\chris\Foosball Detector\First analyzed video\Modified_Track_History_Haas_Klabunde_Moreland_Rue_Tracking_Video_long.json"
# combined_file_path = r"C:\Users\chris\Foosball Detector\First analyzed video\Combined_Tracking_Video_Long.json"
#
# # Load JSON data from the files
# with open(file1_path, 'r') as file:
#     first_data = json.load(file)
#
# with open(file2_path, 'r') as file:
#     second_data = json.load(file)
#
# # Ensure the length of first_data matches the number of frames in second_data
# assert len(first_data) == len(second_data), "Mismatch between the number of entries in the two files."
#
# # Add the `current_ball_location` to each frame in the second data
# for i, frame in enumerate(second_data):
#     frame['current_ball_location'] = first_data[i]
#
# # Save the combined data to a new file
# with open(combined_file_path, 'w') as file:
#     json.dump(second_data, file, indent=4)
#
# print(f"JSON files combined successfully. The result is saved to {combined_file_path}.")
#
#
# print("Hall")
