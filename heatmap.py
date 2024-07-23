import json
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# Load data from a JSON file
file_path = r"C:\Users\chris\Foosball Detector\Vegas_Haas_Klabunde_Moreland_Rue_Long_Video.json"
with open(file_path, 'r') as file:
    data = json.load(file)

# Flexible heat map resolution
resolution_x = 1280
resolution_y = 720

# Initialize a 2D grid for the heat map
heatmap = np.zeros((resolution_y, resolution_x))

# Extract the ball positions (middle of the bounding box) and accumulate the counts
for frame in tqdm(data, desc="Processing frames"):
    for box in frame['boxes']:
        x_min = box['bbox'][0]
        y_min = box['bbox'][1]
        width = box['bbox'][2]
        height = box['bbox'][3]

        # Calculate the center of the bounding box
        x_center = x_min + width / 2
        y_center = y_min + height / 2

        # Convert normalized coordinates to pixel coordinates
        x_pixel = int(x_center * resolution_x)
        y_pixel = int(y_center * resolution_y)

        # Ensure the pixel values are within bounds
        if 0 <= x_pixel < resolution_x and 0 <= y_pixel < resolution_y:
            heatmap[y_pixel, x_pixel] += 1

# Apply logarithmic scaling to the heat map to reduce the effect of highly frequent positions
heatmap_log = np.log1p(heatmap)

# Plotting the heat map
plt.imshow(heatmap_log, cmap='hot', interpolation='nearest', extent=[0, resolution_x, resolution_y, 0])
plt.colorbar()
plt.title('Heat Map of Ball Positions (Middle of Bounding Box)')
plt.xlabel('X Position')
plt.ylabel('Y Position')
plt.show()
