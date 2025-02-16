# import os
# import shutil
# import random
# from tqdm import tqdm
#
# def create_dataset(screenshots_dir, output_dir):
#     # Get the list of all folders in the screenshots directory
#     folders = [f for f in os.listdir(screenshots_dir) if os.path.isdir(os.path.join(screenshots_dir, f))]
#
#     # Initialize a counter for the total number of images
#     total_images = 0
#
#     # Initialize the output_subdir
#     output_subdir = os.path.join(output_dir, '1')
#     os.makedirs(output_subdir, exist_ok=True)
#
#     for folder in tqdm(folders, desc="Processing folders"):
#         # Get the list of all images in the current folder
#         images = [i for i in os.listdir(os.path.join(screenshots_dir, folder)) if i.endswith('.jpg')]
#
#         # Randomly select 1 image
#         selected_images = random.sample(images, 20)
#
#         for image in selected_images:
#             # Increment the total number of images
#             total_images += 1
#
#             # Create a new folder every 500 images
#             if total_images % 500 == 0:
#                 output_subdir = os.path.join(output_dir, str(total_images // 500 + 1))
#                 os.makedirs(output_subdir, exist_ok=True)
#
#             # Copy the selected image to the output directory
#             shutil.copy(os.path.join(screenshots_dir, folder, image), output_subdir)
#
# # Define the directories
# output_dir = "D:\\dataset_random_frames_from_highlight_videos"
# screenshots_dir = "D:\\All-Highlight-Sequences\\screenshots"
#
# # Call the function
# create_dataset(screenshots_dir, output_dir)
#
# yt-dlp --list-formats 'https://www.youtube.com/watch?v=Gz4kJPnpHXg&t=1286s'
#
# yt-dlp -P 'C:/Users/chris/Downloads' -f 612 'https://www.youtube.com/watch?v=Gz4kJPnpHXg&t=1286s'
#
#
# yt-dlp -P 'C:/Users/chris/Downloads' -f 232 'https://www.youtube.com/watch?v=bb8siAHJkmU'
# yt-dlp --list-formats 'https://www.youtube.com/watch?v=bb8siAHJkmU'
#
#
# yt-dlp --list-formats 'https://www.youtube.com/watch?v=OCcbE3Apasg'
# yt-dlp -P 'C:/Users/chris/Downloads' -f 609 'https://www.youtube.com/watch?v=OCcbE3Apasg'
#
#
# yt-dlp --list-formats 'https://www.youtube.com/watch?v=LStf8p91vVo'
# yt-dlp -P 'C:/Users/chris/Downloads' -f 136 'https://www.youtube.com/watch?v=LStf8p91vVo'

yt-dlp --list-formats 'https://www.youtube.com/watch?v=1sBcrUBZxlY'
yt-dlp -P 'C:/Users/chris/Downloads' -f 612 'https://www.youtube.com/watch?v=1sBcrUBZxlY'

yt-dlp --list-formats 'https://www.youtube.com/watch?v=FfT1tNMh0Lg'
yt-dlp -P 'C:/Users/chris/Downloads' -f 612 'https://www.youtube.com/watch?v=FfT1tNMh0Lg'

yt-dlp --list-formats 'https://www.youtube.com/watch?v=QhfrBBLIfvA'
yt-dlp -P 'C:/Users/chris/Downloads' -f 311 'https://www.youtube.com/watch?v=QhfrBBLIfvA'

# video_path = r"C:\Users\chris\Videos\Vegas  ｜  Thomas Haas & Sarah Klabunde vs Brandon Moreland & Sullivan Rue [Gz4kJPnpHXg].mp4"
# annotated_video_path = "output_video.avi"
# final_output_video_path = "Vegas_Haas_Klabunde_Moreland_Rue_Long_Video.mp4"
import json
from collections import defaultdict

# Define a named function for the default factory
track_history_normalized = defaultdict(lambda: [])

# Populate the defaultdict with some example data
track_history_normalized['user1'].extend([1, 2, 3])
track_history_normalized['user2'].extend([4, 5, 6])

# Convert defaultdict to a regular dictionary
track_history_normalized_dict = dict(track_history_normalized)

# Save the dictionary to a JSON file
with open('../../../AppData/Roaming/JetBrains/PyCharmCE2023.2/scratches/track_history_normalized.json', 'w') as file:
    json.dump(track_history_normalized_dict, file)

print("Data has been saved to track_history_normalized.json")

# Load the data from the JSON file
with open('../../../AppData/Roaming/JetBrains/PyCharmCE2023.2/scratches/track_history_normalized.json', 'r') as file:
    loaded_track_history_normalized = json.load(file)

print(loaded_track_history_normalized)
