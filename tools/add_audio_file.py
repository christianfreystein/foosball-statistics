from moviepy.editor import VideoFileClip

# Load the video
video = VideoFileClip(r"C:\Users\chris\Foosball Detector\Deutscher Meister am Kickertisch - Rechtslang Abroller gegen Zieher [aAXxONJDB0A].mp4")

# Trim the video: start at 10 minutes 30 seconds, end at 10 minutes 57 seconds
trimmed_video = video.subclip((4,48), (5,15))

# Save the trimmed video
trimmed_video.write_videofile("trimmed_deutscher_meister_am_kickertisch_video_for_tracking.mp4", codec="libx264", audio_codec="aac")