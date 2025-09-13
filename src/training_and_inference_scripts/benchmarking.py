from ultralytics import YOLO

# # Load a model
model = YOLO("/home/freystec/foosball-statistics/weights/AIF_yolo11n_1280_KD_e40.pt")

# # Validate the model
metrics = model.val(data="/home/freystec/Foosball_Datasets/Second_Prototype_Dataset_with_Leonhart_Topview_Data_cropped/dataset.yaml")


# from ultralytics import YOLO

# # Load a model
# model = YOLO("/home/freystec/foosball-statistics/runs/detect/train5/weights/yolov11n_imgsz640.pt")
model = YOLO("/home/freystec/foosball-statistics/weights/yolov11l_imgsz_640.pt")

metrics = model.val(data="/home/freystec/Foosball_Datasets/Second_Prototype_Dataset_with_Leonhart_Topview_Data_cropped/dataset.yaml")


model = YOLO("/home/freystec/foosball-statistics/weights/yolov11n_imgsz640_distill_bratzdrum_test.pt")
# # Validate the model
metrics = model.val(data="/home/freystec/Foosball_Datasets/Second_Prototype_Dataset_with_Leonhart_Topview_Data_cropped/dataset.yaml")

# from ultralytics import YOLO
model = YOLO("/home/freystec/foosball-statistics/weights/yolov11m_imgsz1280_100e.pt")

metrics = model.val(data="/home/freystec/Foosball_Datasets/Second_Prototype_Dataset_with_Leonhart_Topview_Data_cropped/dataset.yaml")

model = YOLO("repositories/yolo-distiller/runs/detect/train/weights/yolov11n_with_KD_cropped.pt")
             
metrics = model.val(data="/home/freystec/Foosball_Datasets/Second_Prototype_Dataset_with_Leonhart_Topview_Data_cropped/dataset.yaml")
# Load a model
# model = YOLO("/home/freystec/foosball-statistics/runs/detect/train2/weights/yolov11m_Leonhart_Topview.pt")
# model = YOLO("repositories/yolo-distiller/runs/detect/train/weights/yolov11n_with_KD_cropped.pt")
# model = YOLO("/home/freystec/foosball-statistics/weights/yolov11l_imgsz_640.pt")
# Validate the model
# metrics = model.val(data="/home/freystec/Foosball_Datasets/Second_Prototype_Dataset_with_Leonhart_Topview_Data_cropped/dataset.yaml")


# from ultralytics import YOLO

# # Load a model
model = YOLO("/home/freystec/foosball-statistics/runs/detect/train3/weights/yolov11n_imgsz1280_pretrained_with_KD.pt")

# # Validate the model
metrics = model.val(data="/home/freystec/Foosball_Datasets/Second_Prototype_Dataset_with_Leonhart_Topview_Data_cropped/dataset.yaml")


# from ultralytics import YOLO

# # Load a model
# model = YOLO("/home/freystec/foosball-statistics/runs/detect/train3/weights/yolov11n_imgsz1280_pretrained_with_KD.pt")

# # Validate the model
# metrics = model.val(data="/home/freystec/Foosball_Datasets/Second_Prototype_Dataset_with_Leonhart_Topview_Data/dataset.yaml")