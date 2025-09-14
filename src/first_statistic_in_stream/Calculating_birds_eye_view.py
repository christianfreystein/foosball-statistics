import cv2
import numpy as np
import json


def get_homography_and_transform(image_path, json_file):
    """
    Loads an image and four points from a JSON file, calculates a homography,
    and applies a perspective transform to create a birds-eye view.
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

        # Define the destination points for the birds-eye view
        # These coordinates should correspond to a rectangle
        # The dimensions (width, height) are determined by your desired output

        # Let's use the provided dimensions (0,0) (401,0), (401, 650), and (650, 0)
        # Note: The order of destination points must correspond to the source points
        # to ensure the correct transformation.
        # Let's assume the top-left point in your source list corresponds to (0,0),
        # top-right to (650, 0), bottom-right to (650, 401), and bottom-left to (0, 401).
        # We'll use these to create a new destination list.
        # We will need to re-evaluate the destination points. (650,0) and (401,650) is not a rectangle
        # Let's assume we want to map to a standard rectangle.

        width = 617
        height = 1000

        # Re-evaluating the provided destination points
        # The points (0,0), (401,0), (401, 650), and (650, 0) do not form a rectangle.
        # Let's define a correct rectangular destination to map the playing area to.
        dst_points = np.float32([
            [0, 0],  # Top-left
            [width - 1, 0],  # Top-right
            [width - 1, height - 1],  # Bottom-right
            [0, height - 1]  # Bottom-left
        ])

        # Convert the source points to a NumPy array of float32
        src_points = np.float32(src_points)

        # Sort the points to ensure consistent mapping
        # This is a critical step to ensure the perspective transform is correct.
        # We sort them in the order: top-left, top-right, bottom-right, bottom-left.
        src_points_sorted = sort_points(src_points)

        # Calculate the perspective transform matrix (homography)
        M = cv2.getPerspectiveTransform(src_points_sorted, dst_points)

        # Apply the perspective transform to the image
        warped_image = cv2.warpPerspective(image, M, (width, height))

        # Save the transformed image
        output_path = 'transformed_birds_eye_view_edges.jpg'
        cv2.imwrite(output_path, warped_image)
        print(f"Transformed image saved to {output_path}")

    except FileNotFoundError as e:
        print(e)
    except ValueError as e:
        print(e)
    except Exception as e:
        print(f"An error occurred: {e}")


def sort_points(pts):
    """Sorts four points in the order: top-left, top-right, bottom-right, bottom-left."""
    # Based on the sum of x and y coordinates
    # Top-left will have the smallest sum, bottom-right will have the largest sum.
    s = pts.sum(axis=1)

    # Top-left is the point with the smallest sum
    top_left = pts[np.argmin(s)]

    # Bottom-right is the point with the largest sum
    bottom_right = pts[np.argmax(s)]

    # The remaining two points are top-right and bottom-left.
    # The difference of x and y coordinates can help distinguish them.
    # Top-right will have the largest difference (x-y), while bottom-left will have the smallest.
    diff = np.diff(pts, axis=1)

    # Top-right is the point with the smallest difference
    top_right = pts[np.argmin(diff)]

    # Bottom-left is the point with the largest difference
    bottom_left = pts[np.argmax(diff)]

    return np.array([top_left, top_right, bottom_right, bottom_left], dtype="float32")


# --- How to run the script ---
# Make sure your original image and the JSON file are in the same directory.
# Replace 'first_frame.jpg' and 'marked_points_edges.json' with your actual file names.
original_image_file = 'first_frame.jpg'
points_json_file = 'marked_points.json'

get_homography_and_transform(original_image_file, points_json_file)