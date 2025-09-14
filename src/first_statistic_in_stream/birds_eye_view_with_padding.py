import cv2
import numpy as np
import json


def get_homography_and_transform_with_padding(image_path, json_file):
    """
    Performs a perspective transform with padding, mapping the inner four points
    to a smaller rectangle within a larger output image.
    """
    try:
        # Load the image
        image = cv2.imread(image_path)
        if image is None:
            raise FileNotFoundError(f"Image not found at {image_path}")

        # Load the source points from the JSON file
        with open(json_file, 'r') as f:
            src_points = json.load(f)

        # Ensure four points were loaded
        if len(src_points) != 4:
            raise ValueError(f"Expected 4 points, but found {len(src_points)} in {json_file}")

        # Define the padding values
        pad_width = 125
        pad_height = 170

        # Define the destination points for the inner rectangle
        # Let's assume a standard aspect ratio for the playing field.
        # We can calculate the dimensions based on the source points.
        src_points_np = np.float32(src_points)
        src_points_sorted = sort_points(src_points_np)

        # Calculate the width and height of the playing field based on the sorted points
        width = int(np.sqrt(((src_points_sorted[0][0] - src_points_sorted[1][0]) ** 2) + (
                    (src_points_sorted[0][1] - src_points_sorted[1][1]) ** 2)))
        height = int(np.sqrt(((src_points_sorted[0][0] - src_points_sorted[3][0]) ** 2) + (
                    (src_points_sorted[0][1] - src_points_sorted[3][1]) ** 2)))

        # Define the destination points for the inner playing area
        dst_inner_points = np.float32([
            [pad_width, pad_height],  # Top-left inner
            [pad_width + width - 1, pad_height],  # Top-right inner
            [pad_width + width - 1, pad_height + height - 1],  # Bottom-right inner
            [pad_width, pad_height + height - 1]  # Bottom-left inner
        ])

        # Define the dimensions of the final output image
        output_width = width + 2 * pad_width
        output_height = height + 2 * pad_height

        # Calculate the perspective transform matrix (homography)
        M = cv2.getPerspectiveTransform(src_points_sorted, dst_inner_points)

        # Apply the perspective transform to the image, using the new output dimensions
        warped_image = cv2.warpPerspective(image, M, (output_width, output_height))

        # Save the transformed image
        output_path = 'transformed_with_padding.jpg'
        cv2.imwrite(output_path, warped_image)
        print(f"Transformed image with padding saved to {output_path}")

    except FileNotFoundError as e:
        print(e)
    except ValueError as e:
        print(e)
    except Exception as e:
        print(f"An error occurred: {e}")


def sort_points(pts):
    """Sorts four points in the order: top-left, top-right, bottom-right, bottom-left."""
    s = pts.sum(axis=1)
    top_left = pts[np.argmin(s)]
    bottom_right = pts[np.argmax(s)]

    diff = np.diff(pts, axis=1)
    top_right = pts[np.argmin(diff)]
    bottom_left = pts[np.argmax(diff)]

    return np.array([top_left, top_right, bottom_right, bottom_left], dtype="float32")


# --- How to run the script ---
# Make sure your original image and the JSON file are in the same directory.
original_image_file = 'first_frame.jpg'
points_json_file = 'marked_points_edges.json'

get_homography_and_transform_with_padding(original_image_file, points_json_file)