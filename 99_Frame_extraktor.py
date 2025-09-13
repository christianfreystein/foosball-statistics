import os
import cv2
import subprocess
import tkinter as tk
from tkinter import messagebox
from yt_dlp import YoutubeDL

# ================= Hilfsfunktion =================
def hms_to_seconds(hms: str) -> int:
    h, m, s = map(int, hms.split(":"))
    return h*3600 + m*60 + s

# ================= Hauptfunktion =================
def start_process():
    url = entry_url.get()
    start_time = entry_start.get()
    end_time = entry_end.get()
    try:
        frame_step = int(entry_step.get())
        frame_number = int(entry_start_number.get())
    except ValueError:
        messagebox.showerror("Fehler", "Framerate und Startnummer müssen ganze Zahlen sein.")
        return

    if not url or not start_time or not end_time:
        messagebox.showerror("Fehler", "Bitte alle Felder ausfüllen.")
        return

    frames_dir = "frames"
    os.makedirs(frames_dir, exist_ok=True)

    video_full = "video_full.mp4"
    video_cut  = "video_cut.mp4"

    # ================= 1. Video herunterladen =================
    status_label.config(text="Video herunterladen...")
    root.update()

    ydl_opts = {
        'format': 'mp4',
        'outtmpl': video_full,
    }

    try:
        with YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])
    except Exception as e:
        messagebox.showerror("Fehler", f"Download fehlgeschlagen: {e}")
        return

    if not os.path.exists(video_full):
        messagebox.showerror("Fehler", "Download fehlgeschlagen!")
        return

    # ================= 2. Video zuschneiden =================
    start_sec = hms_to_seconds(start_time)
    end_sec   = hms_to_seconds(end_time)
    duration  = end_sec - start_sec

    status_label.config(text="Video zuschneiden...")
    root.update()

    try:
        subprocess.run([
            "ffmpeg", "-y",
            "-i", video_full,
            "-ss", str(start_sec),
            "-t", str(duration),
            "-c", "copy",
            video_cut
        ], check=True)
    except Exception as e:
        messagebox.showerror("Fehler", f"FFmpeg Schnitt fehlgeschlagen: {e}")
        return

    if not os.path.exists(video_cut):
        messagebox.showerror("Fehler", "Video konnte nicht zugeschnitten werden!")
        return

    # ================= 3. Frames extrahieren =================
    status_label.config(text="Frames extrahieren...")
    root.update()

    cap = cv2.VideoCapture(video_cut)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    count = 0
    saved_frame_number = frame_number

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if count % frame_step == 0:
            frame_filename = os.path.join(frames_dir, f"frame_{saved_frame_number}.jpg")
            cv2.imwrite(frame_filename, frame)
            saved_frame_number += 1
            status_label.config(text=f"Frame {saved_frame_number-1} von ~{total_frames//frame_step} gespeichert...")
            root.update()
        count += 1

    cap.release()
    status_label.config(text=f"Fertig! Frames liegen im Ordner: {frames_dir}")
    messagebox.showinfo("Fertig", f"Frames extrahiert: {saved_frame_number - frame_number}")

# ================= GUI =================
root = tk.Tk()
root.title("YouTube Video Frame Extractor")

tk.Label(root, text="YouTube URL:").grid(row=0, column=0, sticky="e")
entry_url = tk.Entry(root, width=50)
entry_url.grid(row=0, column=1)

tk.Label(root, text="Startzeit (HH:MM:SS):").grid(row=1, column=0, sticky="e")
entry_start = tk.Entry(root)
entry_start.grid(row=1, column=1)

tk.Label(root, text="Endzeit (HH:MM:SS):").grid(row=2, column=0, sticky="e")
entry_end = tk.Entry(root)
entry_end.grid(row=2, column=1)

tk.Label(root, text="Framerate (jedes n-te Frame):").grid(row=3, column=0, sticky="e")
entry_step = tk.Entry(root)
entry_step.grid(row=3, column=1)

tk.Label(root, text="Start-Framenummer:").grid(row=4, column=0, sticky="e")
entry_start_number = tk.Entry(root)
entry_start_number.grid(row=4, column=1)

status_label = tk.Label(root, text="Bereit", fg="blue")
status_label.grid(row=5, column=0, columnspan=2)

start_button = tk.Button(root, text="Start", command=start_process)
start_button.grid(row=6, column=0, columnspan=2, pady=10)

root.mainloop()
