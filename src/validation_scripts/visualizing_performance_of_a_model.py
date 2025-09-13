import os
from ultralytics import YOLO

def infer_and_save_annotated_images(model_path, input_folder, output_folder):
    """
    Performs inference on images in a specified folder using a YOLO model
    and saves the annotated images to an output folder.

    Args:
        model_path (str): Path to your trained YOLO model (e.g., 'yolov8n.pt' or 'path/to/best.pt').
        input_folder (str): Path to the folder containing input images.
        output_folder (str): Path to the folder where annotated images will be saved.
    """
    # Load a YOLO model
    try:
        model = YOLO(model_path)
    except Exception as e:
        print(f"Error loading model from {model_path}: {e}")
        print("Please ensure the model path is correct and the model file exists.")
        return

    # Create the output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)
    print(f"Annotated images will be saved to: {os.path.abspath(output_folder)}")

    # Run inference on the input folder and save the results
    # The 'save=True' argument tells Ultralytics to save the annotated images.
    # The 'project' and 'name' arguments help organize the output.
    print(f"Starting inference on images in: {os.path.abspath(input_folder)}")
    results = model.predict(
        source=input_folder,
        save=True,              # Save annotated images
        project=os.path.dirname(output_folder), # Parent directory for runs
        name=os.path.basename(output_folder),   # Specific run name
        exist_ok=True,           # Overwrite previous run with the same name if it exists
        conf = 0.50,
        augment = True,
        show_conf = True
    )

    print("Inference completed and annotated images saved.")
    print("You can find the results in the specified output folder.")


if __name__ == "__main__":
    # --- Configuration ---
    # Replace 'yolov8n.pt' with the path to your trained YOLO model.
    # You can use a pre-trained model like 'yolov8n.pt', 'yolov8s.pt', etc.,
    # or your custom-trained model like 'runs/detect/train/weights/best.pt'.
    YOLO_MODEL_PATH = '/home/freystec/foosball-statistics/runs/detect/train12/weights/best.pt' 

    # Replace 'path/to/your/dataset/images' with the actual path to your image folder.
    # This folder should contain the images you want to inference.
    INPUT_IMAGES_FOLDER = '/home/freystec/foosball_data/new_puppets_cropped' # Example: 'data/test_images'

    # Replace 'path/to/your/output_annotations' with the desired output folder.
    # This is where the annotated images will be saved.
    OUTPUT_ANNOTATIONS_FOLDER = '/home/freystec/foosball_data/new_puppets_cropped/annotated_frames' # Example: 'output/annotated_images'
    # --- End Configuration ---


    # Create a dummy images folder and some dummy images for demonstration if they don't exist
    if not os.path.exists(INPUT_IMAGES_FOLDER):
        print(f"Creating a dummy input folder: {INPUT_IMAGES_FOLDER}")
        os.makedirs(INPUT_IMAGES_FOLDER, exist_ok=True)
        # Create a dummy image file if it doesn't exist for the script to run without error
        try:
            from PIL import Image
            import numpy as np
            dummy_image_path = os.path.join(INPUT_IMAGES_FOLDER, "dummy_image.jpg")
            if not os.path.exists(dummy_image_path):
                img = Image.fromarray(np.random.randint(0, 255, (200, 300, 3), dtype=np.uint8))
                img.save(dummy_image_path)
                print(f"Created dummy image: {dummy_image_path}")
        except ImportError:
            print("Pillow (PIL) not found. Cannot create dummy image. Please ensure your 'images/' folder contains images.")
            print("You can install Pillow with: pip install Pillow")
        except Exception as e:
            print(f"Could not create dummy image: {e}")
            print("Please ensure your 'images/' folder contains images for the script to run.")


    infer_and_save_annotated_images(YOLO_MODEL_PATH, INPUT_IMAGES_FOLDER, OUTPUT_ANNOTATIONS_FOLDER)