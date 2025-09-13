import os
# Set the CUDA_MODULE_LOADING environment variable to LAZY
from ultralytics import YOLO
# os.environ['CUDA_MODULE_LOADING'] = 'LAZY'

# model_path = "/home/freystec/foosball-statistics/weights/yolov8m_imgsz_1280_with_topview_Leonhart_best.engine"
# model_path = "/home/freystec/foosball-statistics/weights/ai_foosball_leonhart_yolo11s_base.pt"
# model_path = "/home/freystec/foosball-statistics/weights/ai_foosball_yolo11n_KD_epoch24.pt"
model_path = "/home/freystec/foosball-statistics/weights/yolov8n_base_ball(slow)_best.pt"
# model_path = "/home/freystec/foosball-statistics/weights/yolov11n_imgsz640_Topview.pt"
video_path = "/home/freystec/foosball-statistics/foosball-videos/Leonhart_clip.mp4"
# model_path = "/home/freystec/foosball-statistics/weights/yolov8n_puppets_first_iteration.pt"

model = YOLO(model_path)

model.predict(video_path, save=False, imgsz=640, batch=1)
