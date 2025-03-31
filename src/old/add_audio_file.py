from moviepy.editor import VideoFileClip

# Load the video
video = VideoFileClip(r"C:\Users\chris\Videos\WS Bonzini 2024 ｜ Open Double ｜ 1⧸32 ｜ Spredeman - Hoffmann vs Wonsyld - Wondyld [OCcbE3Apasg].mp4")

# Trim the video: start at 10 minutes 30 seconds, end at 10 minutes 57 seconds
trimmed_video = video.subclip((1,00), (2,55))

# Save the trimmed video
trimmed_video.write_videofile("D:\Difficult Sequences\Bonzini_Beispiel.mp4", codec="libx264", audio_codec="aac")



