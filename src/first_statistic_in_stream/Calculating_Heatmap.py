import cv2
import numpy as np
import json
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import io


def sort_points(pts):
    """Sorts four points in the order: top-left, top-right, bottom-right, bottom-left."""
    s = pts.sum(axis=1)
    top_left = pts[np.argmin(s)]
    bottom_right = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    top_right = pts[np.argmin(diff)]
    bottom_left = pts[np.argmax(diff)]
    return np.array([top_left, top_right, bottom_right, bottom_left], dtype="float32")


def get_homography(src_points):
    """
    Calculates the homography matrix for a birds-eye view transformation.
    Returns the matrix and the dimensions (width, height) for the transformed view.
    """
    width = 617
    height = 1000

    dst_points = np.float32([
        [0, 0],
        [width - 1, 0],
        [width - 1, height - 1],
        [0, height - 1]
    ])

    src_points_sorted = sort_points(np.float32(src_points))
    M = cv2.getPerspectiveTransform(src_points_sorted, dst_points)

    return M, (width, height)


def transform_points(points, homography_matrix):
    """
    Transforms a list of 2D points using a homography matrix.

    Args:
        points (list of tuples): A list of (x, y) coordinates.
        homography_matrix (np.ndarray): The 3x3 homography matrix.

    Returns:
        np.ndarray: The transformed points as a NumPy array.
    """
    if not points:
        return np.array([])

    points_array = np.array(points, dtype='float32').reshape(-1, 1, 2)
    transformed_points = cv2.perspectiveTransform(points_array, homography_matrix)

    return transformed_points.reshape(-1, 2)


def process_and_transform_data(tracked_data, homography_matrix, frame_width, frame_height):
    """
    Extracts the center of the ball's bounding box, transforms it, and adds it to the data.

    Args:
        tracked_data (list): The list of frames with ball detection data.
        homography_matrix (np.ndarray): The homography matrix.
        frame_width (int): The width of the original frame.
        frame_height (int): The height of the original frame.

    Returns:
        list: The updated data list with transformed ball positions.
    """
    updated_data = []

    for frame in tqdm(tracked_data, desc="Transforming ball positions"):
        bbox = frame.get('current_ball_location')

        if bbox and len(bbox) == 4:
            x_min, y_min, width, height = bbox

            # The coordinates are normalized, so convert to pixel coordinates
            x_center = (x_min + width / 2) * frame_width
            y_center = (y_min + height / 2) * frame_height

            original_point = [(x_center, y_center)]
            transformed_point = transform_points(original_point, homography_matrix)

            frame['transformed_ball_location'] = transformed_point.tolist()[0]
        else:
            frame['transformed_ball_location'] = None

        updated_data.append(frame)

    return updated_data


def transform_image(image_path, homography_matrix, output_dimensions):
    """
    Loads an image, applies a homography transform, and saves the transformed image.

    Args:
        image_path (str): Path to the original image.
        homography_matrix (np.ndarray): The 3x3 homography matrix.
        output_dimensions (tuple): A tuple (width, height) for the transformed image.

    Returns:
        np.ndarray: The transformed image.
    """
    try:
        image = cv2.imread(image_path)
        if image is None:
            raise FileNotFoundError(f"Image not found at {image_path}")

        warped_image = cv2.warpPerspective(image, homography_matrix, output_dimensions)
        return warped_image
    except Exception as e:
        print(f"Error transforming image {image_path}: {e}")
        return None


def generate_heatmap_old(points, output_dimensions, save_path, min_count=2, blur_kernel_size=1, desc_prefix=""):
    """
    Generates and saves a heatmap from a list of points, filtering out rare positions
    and applying a Gaussian blur for a smoother result.
    Returns the generated heatmap grid (before scaling) to be used for overlay.

    Args:
        points (list): List of (x, y) coordinates.
        output_dimensions (tuple): (width, height) of the heatmap grid.
        save_path (str): Path to save the generated heatmap image.
        min_count (int): The minimum number of times a point must be present to be included in the heatmap.
        blur_kernel_size (int): The size of the Gaussian blur kernel.
        desc_prefix (str): Description prefix for the tqdm progress bar.

    Returns:
        np.ndarray: The raw heatmap grid.
    """
    width, height = output_dimensions
    heatmap_grid = np.zeros((height, width), dtype=np.float32)

    # Count occurrences of each point
    point_counts = {}
    for x, y in points:
        x_int, y_int = int(x), int(y)
        if 0 <= x_int < width and 0 <= y_int < height:
            point_counts[(x_int, y_int)] = point_counts.get((x_int, y_int), 0) + 1

    # Filter points based on min_count and populate the grid
    relevant_points = 0
    for (x, y), count in point_counts.items():
        if count >= min_count:
            heatmap_grid[y, x] = count
            relevant_points += 1

    print(
        f"✅ Processed {len(points)} total points. {len(point_counts)} unique points found. {relevant_points} unique points passed the min_count filter.")

    # --- REVISION 1: Apply Gaussian blur for a smoother heatmap ---
    # This turns discrete points into continuous, transparent blobs.
    heatmap_blurred = cv2.GaussianBlur(heatmap_grid, (blur_kernel_size, blur_kernel_size), 0)

    # --- REVISION 2: Use the blurred heatmap for visualization ---
    # Apply logarithmic scaling for visualization
    heatmap_log = np.log1p(heatmap_blurred)

    dpi = 100
    fig_width = width / dpi
    fig_height = height / dpi

    fig, ax = plt.subplots(figsize=(fig_width, fig_height), dpi=dpi)
    ax.imshow(heatmap_log, cmap='hot', interpolation='nearest', extent=[0, width, height, 0])
    ax.set_axis_off()

    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)

    plt.savefig(save_path, dpi=dpi)
    plt.close()
    print(f"✅ Heatmap image saved to {save_path}")

    # Return the blurred heatmap grid for overlaying
    return heatmap_blurred


def extract_points_from_data(tracked_data, key, frame_width, frame_height):
    """
    Extracts ball location points from tracked data in YOLO format
    (x_center, y_center, bbox_width, bbox_height) and returns a list of
    absolute pixel coordinates.
    """
    points = []
    # Use tqdm for progress tracking if the dataset is large
    # The new data format has a "frames" key
    if "frames" in tracked_data:
        frames = tracked_data["frames"]
    else:
        frames = tracked_data

    for frame_data in tqdm(frames, desc="Extracting points"):
        # Check if the key exists and the data is not None
        if "tracked_balls" in frame_data and frame_data["tracked_balls"]:
            for ball_data in frame_data["tracked_balls"]:
                bbox = ball_data.get(key)
                if bbox and len(bbox) == 4:
                    # Assuming the YOLO format is a list/tuple of floats [x_center, y_center, w, h]
                    try:
                        x_center_norm, y_center_norm, _, _ = bbox

                        # Convert normalized coordinates to absolute pixel coordinates
                        x_abs = int(x_center_norm * frame_width)
                        y_abs = int(y_center_norm * frame_height)

                        # Append the absolute point if it's within the frame boundaries
                        if 0 <= x_abs < frame_width and 0 <= y_abs < frame_height:
                            points.append((x_abs, y_abs))
                    except (ValueError, IndexError) as e:
                        print(f"❌ Warning: Could not parse YOLO data from entry {ball_data}. Error: {e}")
                        continue  # Skip to the next entry if parsing fails

    return points


def generate_heatmap(points, output_dimensions, save_path=None, min_count=1, blur_kernel_size=5, desc_prefix=""):
    """
    Generates a more "extreme" heatmap from a list of points.
    If a save_path is provided, it saves the heatmap image to the disk.
    It returns the heatmap grid and the image data buffer.
    """
    width, height = output_dimensions
    heatmap_grid = np.zeros((height, width), dtype=np.float32)

    point_counts = {}
    for x, y in points:
        x_int, y_int = int(x), int(y)
        point_counts[(x_int, y_int)] = point_counts.get((x_int, y_int), 0) + 1

    relevant_points = 0
    for (x, y), count in tqdm(point_counts.items(), desc=f"{desc_prefix}Filtering points"):
        if count >= min_count:
            heatmap_grid[y, x] = count
            relevant_points += 1

    print(
        f"✅ Processed {len(points)} total points. {len(point_counts)} unique points found. {relevant_points} unique points passed the min_count filter.")

    heatmap_blurred = cv2.GaussianBlur(heatmap_grid, (blur_kernel_size, blur_kernel_size), 0)

    # --- Revisions for a more "extreme" heatmap ---
    heatmap_log = np.log2(heatmap_blurred + 1)

    if np.max(heatmap_log) > 0:
        heatmap_normalized = heatmap_log / np.max(heatmap_log)
    else:
        heatmap_normalized = heatmap_log

    dpi = 100
    fig_width = width / dpi
    fig_height = height / dpi

    fig, ax = plt.subplots(figsize=(fig_width, fig_height), dpi=dpi)
    ax.imshow(heatmap_normalized, cmap='hot', interpolation='nearest', extent=[0, width, height, 0])
    ax.set_axis_off()

    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)

    # Use BytesIO to save the figure to an in-memory buffer
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=dpi)
    buf.seek(0)  # Rewind the buffer to the beginning

    # If a save path is provided, save the image to disk
    if save_path:
        plt.savefig(save_path, dpi=dpi)
        print(f"✅ Heatmap image saved to {save_path}")

    plt.close(fig)  # Close the figure to free up memory

    return heatmap_blurred, buf

def overlay_heatmap(image_path, heatmap_data, output_path, alpha=0.9, gamma=0.3):
    """
    Overlays a heatmap on top of a desaturated image using gamma correction.

    Args:
        image_path (str): Path to the original base image.
        heatmap_data (np.ndarray): The heatmap data as a NumPy array.
        output_path (str): Path to save the output image.
        alpha (float): Blending weight for the heatmap.
        gamma (float): Gamma correction value to brighten low-intensity areas.
    """
    try:
        # Load the original base image
        img_original_color = cv2.imread(image_path)
        if img_original_color is None:
            raise FileNotFoundError(f"Original image not found at {image_path}")

        # --- IMPORTANT: Convert heatmap_data to grayscale if it's not already ---
        # This handles cases where the input is a 3-channel image (like a JPG).
        if len(heatmap_data.shape) == 3:
            img_map_grayscale = cv2.cvtColor(heatmap_data, cv2.COLOR_BGR2GRAY)
        else:
            img_map_grayscale = heatmap_data

        # Resize heatmap data to match original image dimensions
        img_map_resized = cv2.resize(img_map_grayscale, (img_original_color.shape[1], img_original_color.shape[0]))

        # --- Core Logic ---
        # Apply gamma correction to the heatmap.
        img_map_gamma_corrected = np.array(255 * (img_map_resized / 255.0) ** gamma, dtype=np.uint8)

        # Apply a colormap to the gamma-corrected heatmap
        heatmap_colored = cv2.applyColorMap(img_map_gamma_corrected, cv2.COLORMAP_HOT)

        # --- Blending ---
        # Convert original image to a desaturated version
        gray_image = cv2.cvtColor(img_original_color, cv2.COLOR_BGR2GRAY)
        desaturated_img = cv2.cvtColor(gray_image, cv2.COLOR_GRAY2BGR)

        # Blend the heatmap with the desaturated image
        super_imposed_img = cv2.addWeighted(heatmap_colored, alpha, desaturated_img, 0.5, 0)

        # --- End of Core Logic ---

        # Save the final image
        cv2.imwrite(output_path, super_imposed_img)
        print(f"✅ Blended image saved to {output_path}")

    except Exception as e:
        print(f"❌ Error overlaying heatmap: {e}")


def save_buffer_to_file(buffer, save_path):
    with open(save_path, 'wb') as f:
        f.write(buffer.getvalue())
    print(f"✅ Heatmap image saved to {save_path}")
# --- Main function to execute the complete workflow ---


def main():
    """
    Main function to execute the complete workflow.
    """
    # Define file paths
    tracked_data_path = r"C:\Users\chris\foosball-statistics\src\first_statistic_in_stream\tracked_results.json"
    screenshot_path = r'C:\Users\chris\foosball-statistics\src\first_statistic_in_stream\first_frame.jpg'

    # Step 1: Load Tracked Ball Data
    try:
        with open(tracked_data_path, 'r') as f:
            tracked_data_raw = json.load(f)
        print("✅ Tracked data loaded successfully.")
    except FileNotFoundError:
        print(f"❌ Error: {tracked_data_path} not found. Please check the path.")
        return

    # Extract original frame dimensions from the metadata
    metadata = tracked_data_raw.get("_metadata_", {})
    if "original_frame_dims" in metadata and len(metadata["original_frame_dims"]) == 2:
        original_frame_width, original_frame_height = metadata["original_frame_dims"]
    else:
        print("❌ Warning: Could not find original frame dimensions in metadata. Using defaults.")
        original_frame_width = 1920
        original_frame_height = 1080

    # Step 2: Extract and Generate Heatmap
    # The `tracked_data_raw` is a dictionary, not a list, so pass it directly
    original_points = extract_points_from_data(tracked_data_raw, 'current_location', original_frame_width,
                                               original_frame_height)
    original_heatmap_output_path = os.path.join(os.path.dirname(tracked_data_path),
                                                'original_ball_position_heatmap.jpg')

    # Step 2: Extract and Generate Heatmap
    # Unpack both the heatmap grid and the buffer
    original_heatmap_grid, original_heatmap_buffer = generate_heatmap(
        original_points,
        (original_frame_width, original_frame_height),
        min_count=1,
        desc_prefix="Original "
    )
    print("✅ Original heatmap generation complete.")

    # Now you have the heatmap grid (a NumPy array)
    # as original_heatmap_grid. You can pass it to overlay_heatmap.

    # Step 3: Overlay the original heatmap on the image with markers
    overlaid_original_path = os.path.join(os.path.dirname(tracked_data_path), 'overlaid_original_heatmap.jpg')

    # Pass the correct variable: original_heatmap_grid (the NumPy array)
    overlay_heatmap(screenshot_path, original_heatmap_grid, overlaid_original_path, alpha=0.90)


if __name__ == '__main__':
    main()
