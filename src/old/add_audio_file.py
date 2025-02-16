from moviepy.editor import VideoFileClip

# Load the video
video = VideoFileClip(r"C:\Users\chris\foosball-statistics\Second_Prototype_without_Impossible_Spredeman_Hoffman_Wonsyld_Wonsyld.mp4")

# Trim the video: start at 10 minutes 30 seconds, end at 10 minutes 57 seconds
trimmed_video = video.subclip((0,20), (1,1))

# Save the trimmed video
trimmed_video.write_videofile("D:\Difficult Sequences\Bonzini_Beispiel.mp4", codec="libx264", audio_codec="aac")



