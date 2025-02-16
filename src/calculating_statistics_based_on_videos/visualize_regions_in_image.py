import cv2
import numpy as np
import matplotlib.pyplot as plt


def draw_regions(image, regions):
    """
    Draws the foosball table regions on the given image.

    Parameters:
        image (numpy array): The image on which to draw.
        regions (list): List of tuples (quad, region_name), where
                        quad is a list of (x, y) points defining a region.
    """
    for quad, region_name in regions:
        # Convert to integer coordinates and reshape properly
        quad = np.array(quad, np.int32).reshape((-1, 1, 2))

        # Draw the region
        cv2.polylines(image, [quad], isClosed=True, color=(0, 255, 0), thickness=2)

        # Compute text position (center of the quadrilateral)
        x_coords = [point[0] for point in quad[:, 0]]  # Extract x coordinates
        y_coords = [point[1] for point in quad[:, 0]]  # Extract y coordinates
        cx, cy = int(np.mean(x_coords)), int(np.mean(y_coords))  # Find center

        # Put the region name on the image
        cv2.putText(image, region_name, (cx - 40, cy), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (255, 255, 255), 2, cv2.LINE_AA)

    return image


# Load a test image or create a blank one
image_path = r"C:\Users\chris\foosball-statistics\random_frame.jpg"
image = cv2.imread(image_path)

# Ensure the image was loaded correctly
if image is None:
    raise FileNotFoundError(f"Error: Unable to load image from {image_path}")

# Define foosball regions (using your provided coordinates)
regions = [
    ([(508, 64), (835, 70), (845, 184), (499, 173)], "Left 2"),
    ([(499, 173), (845, 184), (847, 242), (496, 239)], "Right 0"),
    ([(496, 239), (847, 242), (849, 307), (494, 304)], "Left 1"),
    ([(494, 304), (849, 307), (857, 377), (485, 373)], "Right 1"),
    ([(485, 373), (857, 377), (863, 451), (475, 447)], "Left 0"),
    ([(475, 447), (863, 451), (871, 578), (467, 571)], "Right 2"),
]

# Draw the regions on the image
image_with_regions = draw_regions(image, regions)

# Display the image using Matplotlib
plt.figure(figsize=(12, 7))
plt.imshow(cv2.cvtColor(image_with_regions, cv2.COLOR_BGR2RGB))
plt.title("Foosball Table Regions")
plt.axis("off")  # Hide axis
plt.show()

# Save the result
output_path = r"C:\Users\chris\foosball-statistics\foosball_regions.jpg"
cv2.imwrite(output_path, image_with_regions)
print(f"Saved output image to {output_path}")

