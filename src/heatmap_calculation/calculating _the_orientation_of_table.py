import cv2
import numpy as np

def calculate_rotation_angle(image_path):
    """
    Calculates the rotation angle of the foosball table based on a heatmap.
    
    Args:
        image_path (str): The file path to the heatmap image.
        
    Returns:
        float: The rotation angle in degrees needed to make the table horizontal.
               Returns None if no lines are detected.
    """
    # 1. Load the image and convert to grayscale
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"Error: Could not load image at {image_path}")
        return None

    # 2. Apply a threshold to isolate the high-intensity points (ball positions)
    # The threshold value (e.g., 50) may need to be adjusted based on your image's brightness.
    _, thresh = cv2.threshold(img, 50, 255, cv2.THRESH_BINARY)
    
    # 3. Use Canny edge detection for better line detection
    edges = cv2.Canny(thresh, 50, 150, apertureSize=3)
    
    # 4. Use the Probabilistic Hough Transform to detect lines
    # The parameters (e.g., 50, 20, 10) are fine-tuned for this use case.
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, 50, minLineLength=20, maxLineGap=10)
    
    if lines is None:
        print("No lines were detected in the image.")
        return None
        
    angles = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        # Calculate angle in radians
        angle_rad = np.arctan2(y2 - y1, x2 - x1)
        # Convert to degrees
        angle_deg = np.rad2deg(angle_rad)
        
        # Adjust angles to be between -90 and +90 degrees for consistency
        if angle_deg < 0:
            angle_deg += 180
            
        angles.append(angle_deg)
    
    # Filter for dominant angles (lines that are mostly horizontal)
    # We look for angles close to 0 or 180 (which is 0 after our adjustment)
    dominant_angles = [angle for angle in angles if 80 < angle < 100 or -10 < angle < 10 or 170 < angle < 190]
    
    # 5. Find the median angle to get a robust rotation value
    if dominant_angles:
        # The median is a good choice as it is less affected by outliers
        median_angle = np.median(dominant_angles)
        
        # The target orientation is 90 degrees (vertical), so we subtract it to get the rotation angle
        rotation_angle = 90 - median_angle if median_angle > 45 else -median_angle
        
        return rotation_angle
    else:
        print("Could not find a dominant horizontal or vertical orientation.")
        return None

# --- Example Usage ---
if __name__ == "__main__":
    image_file = '/home/freystec/foosball-statistics/src/heatmap_calculation/SiMo_yolov11n_imgsz_640_long_tracked_data.jpg'
    
    rotation_angle = calculate_rotation_angle(image_file)
    
    if rotation_angle is not None:
        print(f"The calculated rotation angle is: {rotation_angle:.2f} degrees")
        
        # Now, you can rotate the original image using OpenCV
        img_orig = cv2.imread(image_file)
        if img_orig is not None:
            # Get the image dimensions
            h, w = img_orig.shape[:2]
            # Calculate the rotation matrix
            center = (w // 2, h // 2)
            M = cv2.getRotationMatrix2D(center, rotation_angle, 1.0)
            
            # Perform the rotation
            rotated_img = cv2.warpAffine(img_orig, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
            
            # Display images to show the result
            # NOTE: Use this only in a local environment where a display window is available
            cv2.imshow('Original Image', img_orig)
            cv2.imshow('Rotated Image', rotated_img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()