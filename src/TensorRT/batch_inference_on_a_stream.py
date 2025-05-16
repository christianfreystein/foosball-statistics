import cv2
import threading
import queue
import time
from ultralytics import YOLO

model_path = "/home/freystec/foosball-statistics/weights/yolov8m_imgsz_1280_with_topview_Leonhart_best.engine"
# model_path = "/home/freystec/foosball-statistics/weights/ai_foosball_leonhart_yolo11s_base.pt"
# model_path = "/home/freystec/foosball-statistics/weights/ai_foosball_leonhart_yolo11s_base.onnx"
model_path = "/home/freystec/foosball-statistics/weights/yolov8n_base_ball(slow)_best.pt"
video_path = "/home/freystec/foosball-statistics/foosball-videos/Leonhart_clip.mp4"

# Initialize YOLO model
model = YOLO(model_path) # You can change this to yolov8s.pt or your custom model

# Config
BATCH_SIZE = 16
VIDEO_SOURCE = video_path  # Can be a video path or camera index
FRAME_QUEUE_MAXSIZE = 64

# Metrics
dropped_frames = 0
shown_frames = 0
start_display_time = None
end_display_time = None
total_video_frames = 0
video_fps = 0.0

# Thread-safe queue for frames
frame_queue = queue.Queue(maxsize=FRAME_QUEUE_MAXSIZE)

# Flag to stop threads gracefully
stop_flag = threading.Event()


def frame_reader():
    global total_video_frames, video_fps

    cap = cv2.VideoCapture(VIDEO_SOURCE)
    if not cap.isOpened():
        print("Error: Could not open video source.")
        stop_flag.set()
        return

    # Get total frames and FPS
    total_video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_fps = cap.get(cv2.CAP_PROP_FPS)

    while not stop_flag.is_set():
        ret, frame = cap.read()
        if not ret:
            print("End of stream or error.")
            stop_flag.set()
            break

        try:
            frame_queue.put(frame, timeout=1)
        except queue.Full:
            pass  # Drop frames if queue is full

    cap.release()


def batch_inference():
    """Performs batch inference with YOLO on batches of frames and displays at ~60 FPS."""
    frame_duration = 1.0 / 30.0  # Target: 60 FPS

    while not stop_flag.is_set():
        batch = []

        # 🧠 Drop older frames if backlog exists
        global dropped_frames, shown_frames, start_display_time, end_display_time

        while not frame_queue.empty() and frame_queue.qsize() > BATCH_SIZE:
            try:
                frame_queue.get_nowait()
                dropped_frames += 1
            except queue.Empty:
                break


        # Collect fresh BATCH_SIZE frames
        while len(batch) < BATCH_SIZE:
            try:
                frame = frame_queue.get(timeout=1)
                batch.append(frame)
            except queue.Empty:
                if stop_flag.is_set():
                    break

        if not batch:
            continue

        # Run batch inference
        results = model(batch)

        # Display results at ~60 FPS
        for result in results:
            # Track first/last display time
            if start_display_time is None:
                start_display_time = time.time()

            end_display_time = time.time()
            shown_frames += 1

            annotated = result.plot()

            start_time = time.time()
            cv2.imshow("YOLO Real-Time Inference", annotated)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                stop_flag.set()
                break

            elapsed = time.time() - start_time
            delay = frame_duration - elapsed
            if delay > 0:
                time.sleep(delay)


if __name__ == "__main__":
    try:
        # Start reader thread
        reader_thread = threading.Thread(target=frame_reader, daemon=True)
        reader_thread.start()

        # Run batch inference loop
        batch_inference()

        

    except KeyboardInterrupt:
        print("Interrupted by user.")
    finally:
        stop_flag.set()
        cv2.destroyAllWindows()

        # Print metrics
        print("\n--- Video Processing Summary ---")
        print(f"Total frames in video:      {total_video_frames}")
        print(f"Video FPS (from file):      {video_fps:.2f}")
        print(f"Real video duration:        {total_video_frames / video_fps:.2f} seconds")
        print(f"Displayed frames:           {shown_frames}")
        print(f"Dropped frames (in queue):  {dropped_frames}")
        if start_display_time and end_display_time:
            print(f"Actual display time:        {end_display_time - start_display_time:.2f} seconds")
            print(f"Achieved display FPS:       {shown_frames / (end_display_time - start_display_time):.2f}")
        print("--------------------------------")
