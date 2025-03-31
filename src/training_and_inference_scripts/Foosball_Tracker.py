import numpy as np
import supervision as sv
from ultralytics import YOLO
from collections import deque
import cv2


class BallAnnotator:
    def __init__(
            self,
            radius: int,
            buffer_size: int = 5,
            thickness: int = 2
    ):
        self.color_palette = sv.ColorPalette.from_matplotlib('jet', buffer_size)
        self.buffer = deque(maxlen=buffer_size)
        self.radius = radius
        self.thickness = thickness

    def interpolate_radius(self, i: int, max_i: int) -> int:
        if max_i == 1:
            return self.radius
        return int(1 + i * (self.radius - 1) / (max_i - 1))

    def annotate(
            self,
            frame: np.ndarray,
            detections: sv.Detections
    ) -> np.ndarray:
        xy = detections.get_anchors_coordinates(
            sv.Position.BOTTOM_CENTER).astype(int)
        self.buffer.append(xy)

        for i, xy in enumerate(self.buffer):
            color = self.color_palette.by_idx(i)
            interpolated_radius = self.interpolate_radius(i, len(self.buffer))

            for center in xy:
                frame = cv2.circle(
                    img=frame,
                    center=tuple(center),
                    radius=interpolated_radius,
                    color=color.as_bgr(),
                    thickness=self.thickness
                )
        return frame


class BallTracker:
    def __init__(self, buffer_size: int = 10):
        self.buffer = deque(maxlen=buffer_size)

    def update(self, detections: sv.Detections) -> sv.Detections:
        xy = detections.get_anchors_coordinates(sv.Position.CENTER)
        self.buffer.append(xy)

        if len(detections) == 0:
            return detections

        centroid = np.mean(np.concatenate(self.buffer), axis=0)
        distances = np.linalg.norm(xy - centroid, axis=1)
        index = np.argmin(distances)
        return detections[[index]]


model = YOLO(r"C:\Users\chris\foosball-statistics\runs\detect\train5\weights\best_run5.pt")
source_path = (r"C:\Users\chris\foosball-statistics\Leonhart_clip - Kopie.mp4")
target_path = (r"C:\Users\chris\foosball-statistics\Leonhart_clip_test.mp4")
video_info = sv.VideoInfo.from_video_path(source_path)
frame_generator = sv.get_video_frames_generator(source_path)
w, h = video_info.width, video_info.height

# annotator = sv.TriangleAnnotator(
#     color=sv.Color.from_hex('#FF1493'),
#     height=20,
#     base=25
# )
annotator = BallAnnotator(20,30,4)
tracker = BallTracker()


with sv.VideoSink(target_path, video_info=video_info) as sink:
    for frame in frame_generator:
        results = model(frame)[0]
        detections = sv.Detections.from_ultralytics(results)
        detections = tracker.update(detections)
        frame = annotator.annotate(frame, detections)
        sink.write_frame(frame)
