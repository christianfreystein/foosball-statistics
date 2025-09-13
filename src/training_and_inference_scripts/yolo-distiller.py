from ultralytics import YOLO

teacher_model = YOLO("/home/freystec/foosball-statistics/weights/yolov11l_imgsz_640.pt")

student_model = YOLO("/home/freystec/foosball-statistics/runs/detect/train4/weights/best.pt")

student_model.train(
    data="/home/freystec/Foosball_Datasets/Second_Prototype_Dataset_with_Leonhart_Topview_Data_cropped/dataset.yaml",
    teacher=teacher_model.model, # None if you don't wanna use knowledge distillation
    distillation_loss="cwd",
    epochs=100,
    batch=-1,
    workers=0,
    exist_ok=True,
)