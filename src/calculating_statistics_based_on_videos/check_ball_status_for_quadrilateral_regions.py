import json
from tqdm import tqdm


def read_json(file_path):
    with open(file_path, 'r') as file:
        return json.load(file)


def write_json(data, file_path):
    with open(file_path, 'w') as file:
        json.dump(data, file, indent=4)


def is_point_in_triangle(px, py, ax, ay, bx, by, cx, cy):
    """Check if point (px, py) is inside the triangle (ax, ay), (bx, by), (cx, cy)"""

    def sign(x1, y1, x2, y2, x3, y3):
        return (x1 - x3) * (y2 - y3) - (x2 - x3) * (y1 - y3)

    d1 = sign(px, py, ax, ay, bx, by)
    d2 = sign(px, py, bx, by, cx, cy)
    d3 = sign(px, py, cx, cy, ax, ay)

    has_neg = (d1 < 0) or (d2 < 0) or (d3 < 0)
    has_pos = (d1 > 0) or (d2 > 0) or (d3 > 0)

    return not (has_neg and has_pos)


def is_point_in_quadrilateral(px, py, quad):
    """Check if point (px, py) is inside the quadrilateral"""
    (x1, y1), (x2, y2), (x3, y3), (x4, y4) = quad
    return (is_point_in_triangle(px, py, x1, y1, x2, y2, x3, y3) or
            is_point_in_triangle(px, py, x1, y1, x3, y3, x4, y4))


def find_region(x, y, regions):
    """Find the region where the ball is located"""
    for quad, region_name in regions:
        if is_point_in_quadrilateral(x, y, quad):
            return region_name
    return "Outside"


def add_ball_status(data, image_width, image_height, regions):
    """Updates each frame with the correct ball status based on x and y coordinates."""
    for frame in tqdm(data, desc="Processing frames"):
        if 'current_ball_location' in frame:
            # Extract normalized coordinates
            normalized_x, normalized_y, _, _ = frame['current_ball_location']

            # Convert to pixel values
            x_pixel = normalized_x * image_width
            y_pixel = normalized_y * image_height

            # Determine ball status
            ball_status = find_region(x_pixel, y_pixel, regions)
            frame['ball_status'] = ball_status

    return data


# Define file paths
input_file_path = r"/Second_Prototype_without_Impossible_Westermann_Bade_Hoffmann_Spredeman.json"
output_file_path = r"/Second_Prototype_without_Impossible_Westermann_Bade_Hoffmann_Spredeman_with_ball_status.json"

# Image dimensions
image_width = 1280
image_height = 720

# Define regions using (x, y) tuples instead of a flat list
regions = [
    ([(508, 64), (835, 70), (845, 184), (499, 173)], "Left 2"),
    ([(499, 173), (845, 184), (847, 242), (496, 239)], "Right 0"),
    ([(496, 239), (847, 242), (849, 307), (494, 304)], "Left 1"),
    ([(494, 304), (849, 307), (857, 377), (485, 373)], "Right 1"),
    ([(485, 373), (857, 377), (863, 451), (475, 447)], "Left 0"),
    ([(475, 447), (863, 451), (871, 578), (467, 571)], "Right 2"),
]

# Load JSON data
data = read_json(input_file_path)

# Update ball status
updated_data = add_ball_status(data, image_width, image_height, regions)

# Save updated data
write_json(updated_data, output_file_path)

print(f"JSON file updated successfully. The result is saved to {output_file_path}.")


# Define regions as quadrilaterals (each has 4 points)
'''
(x1, y1) ------------- (x2, y2)
     |                     |
     |                     |
(x4, y4) ------------- (x3, y3)
'''