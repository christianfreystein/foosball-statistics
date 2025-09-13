from PIL import Image
import numpy as np

def analyze_image_horizontal_cut(image_path):
    """
    Reads an image and performs a horizontal cut, returning the sum of vertical values.

    Args:
        image_path (str): The path to the image file.

    Returns:
        np.ndarray: A NumPy array containing the sum of pixel values for each
                    horizontal row, or None if an error occurs.
    """
    try:
        # Open the image and convert it to a grayscale NumPy array
        img = Image.open(image_path)
        img_array = np.array(img.convert('L'))

        # Sum the pixel values along the horizontal axis (each row)
        row_sums = np.sum(img_array, axis=1)

        return row_sums

    except FileNotFoundError:
        print(f"Error: The file at {image_path} was not found.")
        return None
    except Exception as e:
        print(f"An error occurred: {e}")
        return None
    

import matplotlib.pyplot as plt

def save_plot(data_array, output_plot_path):
    """
    Generates and saves a plot from a given NumPy array.

    Args:
        data_array (np.ndarray): The data to plot.
        output_plot_path (str): The path to save the output plot (e.g., 'plot.png').
    """
    try:
        # Create a new plot
        plt.figure(figsize=(10, 6))
        plt.plot(data_array, color='red')
        plt.title('Sum of Pixel Values for Each Horizontal Row')
        plt.xlabel('Row Index (y-axis)')
        plt.ylabel('Total Pixel Value (Brightness)')
        plt.grid(True)

        # Save the plot to a file
        plt.savefig(output_plot_path)
        print(f"Plot saved successfully to {output_plot_path}")

    except Exception as e:
        print(f"An error occurred while saving the plot: {e}")


# --- Example usage ---
input_image_file = '/home/freystec/foosball-statistics/src/heatmap_calculation/Struth_Pelzer_Hüwe_Weiss_Detections.jpg'
output_plot_file = '/home/freystec/foosball-statistics/src/heatmap_calculation/Struth_Pelzer_Hüwe_Weiss_Detections_analysis_plot.png'

# 1. Perform the analysis and get the data
analysis_results = analyze_image_horizontal_cut(input_image_file)

# 2. Check if the analysis was successful and then save the plot
if analysis_results is not None:
    save_plot(analysis_results, output_plot_file)