import cv2
import numpy as np

# Load the image
input_image_path = r"C:\Users\chris\Foosball Detector\test_analyzed_video\test_bild.png"
image = cv2.imread(input_image_path)

# Output image size
output_width = 1280
output_height = 720

# Color mapping for ball status
status_color_mapping = {
    "Left 2": (0, 0, 255),    # Red
    "Right 0": (0, 255, 0),   # Green
    "Left 1": (0, 165, 255),  # Orange
    "Right 1": (0, 255, 255), # Yellow
    "Left 0": (255, 0, 255),  # Magenta
    "Right 2": (255, 255, 0), # Cyan
    "Outside": (128, 128, 128) # Gray
}

# y_areas definition
y_areas = {
    (61, 162): "Left 2",
    (162, 209): "Right 0",
    (209, 259): "Left 1",
    (259, 331): "Right 1",
    (331, 415): "Left 0",
    (415, 570): "Right 2"
}

# Resize the image to the desired size
resized_image = cv2.resize(image, (output_width, output_height))

# Draw upper and lower borders for each area
for (y1, y2), status in y_areas.items():
    color = status_color_mapping.get(status, (255, 255, 255))  # Default to white if status is not found
    # Calculate the line y-coordinates
    upper_y = int(y1)
    lower_y = int(y2)-2
    # Draw upper border line
    cv2.line(resized_image, (0, upper_y), (output_width, upper_y), color, 2)  # Upper border line
    # Draw lower border line
    cv2.line(resized_image, (0, lower_y), (output_width, lower_y), color, 2)  # Lower border line
    # Adjust text position to be in the middle of the area
    text_x = output_width // 2 - 100
    text_y = (upper_y + lower_y) // 2
    cv2.putText(resized_image, status, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, cv2.LINE_AA)

# Display the image
cv2.imshow("Visualized Areas", resized_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Save the image
output_image_path = r"C:\Users\chris\Foosball Detector\test_analyzed_video\Visualized_Areas.png"
cv2.imwrite(output_image_path, resized_image)
