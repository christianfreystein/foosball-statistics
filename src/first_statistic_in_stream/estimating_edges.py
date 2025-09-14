import cv2
import json
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk

# Global variables to store points and the JSON filename
points = []
json_filename = 'marked_points.json'


def save_points_to_json():
    """Saves the marked points to a JSON file."""
    with open(json_filename, 'w') as f:
        json.dump(points, f, indent=4)
    print(f"Points saved to {json_filename}")


def on_click(event):
    """Event handler for mouse clicks on the canvas."""
    x, y = event.x, event.y
    points.append((x, y))
    print(f"Point marked at: ({x}, {y})")

    # Draw a circle on the canvas to visually confirm the point
    canvas.create_oval(x - 3, y - 3, x + 3, y + 3, fill='red', outline='red')


def open_video_and_show_first_frame():
    """Opens a video file and displays its first frame in the GUI."""
    global photo, video_capture, first_frame_image

    file_path = filedialog.askopenfilename(
        title="Select a video file",
        filetypes=(("Video files", "*.mp4 *.avi *.mov"), ("All files", "*.*"))
    )

    if not file_path:
        return

    video_capture = cv2.VideoCapture(file_path)
    success, frame = video_capture.read()

    if success:
        # Convert the frame from BGR to RGB
        first_frame_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Save the first frame as an image file
        cv2.imwrite('first_frame.jpg', frame)
        print("First frame saved as first_frame.jpg")

        # Convert the image for Tkinter display
        image_pil = Image.fromarray(first_frame_image)
        photo = ImageTk.PhotoImage(image_pil)

        # Update the canvas with the new image
        canvas.config(width=photo.width(), height=photo.height())
        canvas.create_image(0, 0, anchor=tk.NW, image=photo)

        # Bind the click event to the canvas
        canvas.bind('<Button-1>', on_click)

    else:
        print("Failed to read the first frame of the video.")

    video_capture.release()


# --- Main GUI setup ---
root = tk.Tk()
root.title("Video Playable Area Marker")

# Create a frame for buttons and the canvas
main_frame = tk.Frame(root)
main_frame.pack(padx=10, pady=10)

# Create buttons
open_button = tk.Button(main_frame, text="Open Video", command=open_video_and_show_first_frame)
open_button.pack(side=tk.LEFT, padx=5)

save_button = tk.Button(main_frame, text="Save Points", command=save_points_to_json)
save_button.pack(side=tk.LEFT, padx=5)

# Create a canvas to display the image and draw points
canvas = tk.Canvas(root, bg='gray')
canvas.pack()

# Start the Tkinter event loop
root.mainloop()