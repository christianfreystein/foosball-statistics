import json
import numpy as np
import matplotlib.pyplot as plt
import cv2
from tqdm import tqdm

def find_rotation_angle(img_array):
    """
    Finds the rotation angle of the table by detecting the dominant lines
    in the heatmap using a Hough Transform.
    
    Args:
        img_array (np.ndarray): The 2D numpy array of the heatmap.
        
    Returns:
        float: The rotation angle in degrees.
    """
    # Convert the floating point heatmap array to an 8-bit integer format for OpenCV
    # We first normalize it to the 0-255 range.
    norm_img = cv2.normalize(img_array, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
    
    # Apply a binary threshold to isolate the brightest points
    _, thresh = cv2.threshold(norm_img, 50, 255, cv2.THRESH_BINARY)
    
    # Apply Canny edge detection for better line detection
    edges = cv2.Canny(thresh, 50, 150, apertureSize=3)
    
    # Use the Probabilistic Hough Transform
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 50, minLineLength=20, maxLineGap=10)
    
    if lines is None:
        print("No lines were detected in the image. Returning 0 degrees.")
        return 0.0

    angles = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        # Calculate angle and adjust to be between -90 and +90 degrees
        angle_rad = np.arctan2(y2 - y1, x2 - x1)
        angle_deg = np.rad2deg(angle_rad)
        if angle_deg > 90:
            angle_deg -= 180
        elif angle_deg < -90:
            angle_deg += 180
        angles.append(angle_deg)
    
    # Filter for the most dominant angle
    if angles:
        # Find the most frequent angle range
        hist, bin_edges = np.histogram(angles, bins=np.arange(-90, 91, 1))
        dominant_bin = np.argmax(hist)
        dominant_angle = (bin_edges[dominant_bin] + bin_edges[dominant_bin+1]) / 2
        
        return dominant_angle
    else:
        return 0.0

def rotate_image(image, angle):
    """Rotates an image array around its center."""
    h, w = image.shape[:2]
    center = (w / 2, h / 2)
    
    # Get the rotation matrix
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    
    # Perform the rotation, preserving the image size and filling borders with black
    rotated_image = cv2.warpAffine(image, rotation_matrix, (w, h), borderValue=(0,0,0))
    return rotated_image

# --- Main Script ---

if __name__ == "__main__":
    # --- Configuration ---
    file_path = "/home/freystec/Spredeman_Hoffmann_vs_Wonsyld_Wondy_tracked_data.json"
    save_dir = "/home/freystec/foosball-statistics/src/heatmap_calculation/"
    resolution_x = 1280
    resolution_y = 720

    # --- Step 1: Data Loading ---
    print("Loading data...")
    try:
        with open(file_path, 'r') as file:
            data = json.load(file)
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
        exit()

    # --- Step 2: Heatmap Creation (Raw Counts) ---
    print("Creating raw heatmap...")
    heatmap = np.zeros((resolution_y, resolution_x))
    for frame in tqdm(data, desc="Processing frames"):
        for box in frame['boxes']:
            x_min, y_min, width, height = box['bbox']
            x_center = x_min + width / 2
            y_center = y_min + height / 2
            x_pixel = int(x_center * resolution_x)
            y_pixel = int(y_center * resolution_y)

            if 0 <= x_pixel < resolution_x and 0 <= y_pixel < resolution_y:
                heatmap[y_pixel, x_pixel] += 1
    
    # Save the raw heatmap visualization
    heatmap_normalized = heatmap / np.max(heatmap)
    plt.imsave(save_dir + 'heatmap_raw_counts.png', heatmap_normalized, cmap='hot')
    print(f"Saved raw heatmap to {save_dir}heatmap_raw_counts.png")

    # --- Step 3: Logarithmic Scaling ---
    print("Applying logarithmic scaling...")
    heatmap_log = np.log1p(heatmap)

    # Save the logarithmically scaled heatmap visualization
    plt.imsave(save_dir + 'heatmap_log_scaled.png', heatmap_log, cmap='hot')
    print(f"Saved log-scaled heatmap to {save_dir}heatmap_log_scaled.png")

    # --- Step 4: Hough Transform and Rotation Calculation ---
    print("Calculating rotation angle...")
    rotation_angle = find_rotation_angle(heatmap_log)
    print(f"The detected rotation angle is: {rotation_angle:.2f} degrees")

    # --- Step 5: Rotate the Heatmap and Save the Final Image ---
    print("Rotating and saving final corrected heatmap...")
    # Rotate the log-scaled heatmap
    rotated_heatmap = rotate_image(heatmap_log, rotation_angle)
    
    # Plot and save the final corrected image
    plt.imsave(save_dir + 'heatmap_corrected.png', rotated_heatmap, cmap='hot')
    print(f"Saved final corrected heatmap to {save_dir}heatmap_corrected.png")
    
    print("Process completed successfully.")