import cv2
import threading
import queue
import time
from ultralytics import YOLO

# === CONFIG ===
BATCH_SIZE = 16
VIDEO_SOURCE = "/home/freystec/foosball-statistics/foosball-videos/Leonhart_clip.mp4"
MODEL_PATH = "/home/freystec/foosball-statistics/weights/yolov8n_base_ball(slow)_best.pt"
FRAME_QUEUE_MAXSIZE = 64
TARGET_DISPLAY_FPS = 60
DISPLAY_FRAME_DURATION = 1.0 / TARGET_DISPLAY_FPS

# === GLOBALS ===
frame_queue = queue.Queue(maxsize=FRAME_QUEUE_MAXSIZE)
latest_annotated_frames = []
latest_lock = threading.Lock()

stop_flag = threading.Event()
dropped_batches = 0
processed_batches = 0
shown_frames = 0
start_display_time = None
end_display_time = None
total_video_frames = 0
video_fps = 0.0

# === LOAD MODEL ===
model = YOLO(MODEL_PATH)


def frame_reader():
    global total_video_frames, video_fps

    cap = cv2.VideoCapture(VIDEO_SOURCE)
    if not cap.isOpened():
        print("❌ Could not open video source.")
        stop_flag.set()
        return

    total_video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_fps = cap.get(cv2.CAP_PROP_FPS)

    while not stop_flag.is_set():
        ret, frame = cap.read()
        if not ret:
            break

        try:
            frame_queue.put(frame, timeout=1)
        except queue.Full:
            pass  # Drop frame silently

    cap.release()


def batch_inference():
    global dropped_batches, processed_batches, latest_annotated_frames

    while not stop_flag.is_set():
        batch = []

        # Drop stale frames if too many are in the queue
        while not frame_queue.empty() and frame_queue.qsize() > BATCH_SIZE * 2:
            try:
                frame_queue.get_nowait()
                dropped_batches += 1
            except queue.Empty:
                break

        while len(batch) < BATCH_SIZE and not stop_flag.is_set():
            try:
                frame = frame_queue.get(timeout=1)
                batch.append(frame)
            except queue.Empty:
                break

        if not batch:
            continue

        results = model(batch)
        processed_batches += 1

        # Store annotated frames for display
        with latest_lock:
            latest_annotated_frames = [res.plot() for res in results]


def display_loop():
    global start_display_time, end_display_time, shown_frames

    frame_index = 0

    while not stop_flag.is_set():
        with latest_lock:
            if not latest_annotated_frames:
                continue
            frame = latest_annotated_frames[frame_index % len(latest_annotated_frames)]

        if start_display_time is None:
            start_display_time = time.time()

        cv2.imshow("YOLO Real-Time Batch Display", frame)
        shown_frames += 1

        if cv2.waitKey(1) & 0xFF == ord('q'):
            stop_flag.set()
            break

        frame_index += 1
        time.sleep(DISPLAY_FRAME_DURATION)

    end_display_time = time.time()


if __name__ == "__main__":
    try:
        reader_thread = threading.Thread(target=frame_reader, daemon=True)
        inference_thread = threading.Thread(target=batch_inference, daemon=True)

        reader_thread.start()
        inference_thread.start()

        display_loop()

    except KeyboardInterrupt:
        stop_flag.set()

    finally:
        stop_flag.set()
        cv2.destroyAllWindows()

        print("\n--- Batch Inference Video Summary ---")
        print(f"Total frames in video:        {total_video_frames}")
        print(f"Video FPS (from file):        {video_fps:.2f}")
        print(f"Real video duration:          {total_video_frames / video_fps:.2f} seconds")
        print(f"Processed batches:            {processed_batches}")
        print(f"Dropped batches (frames):     {dropped_batches}")
        print(f"Displayed frames:             {shown_frames}")
        if start_display_time and end_display_time:
            duration = end_display_time - start_display_time
            print(f"Actual display time:          {duration:.2f} seconds")
            print(f"Achieved display FPS:         {shown_frames / duration:.2f}")
        print("--------------------------------------")
