from ultralytics import YOLO


def main():
    model = YOLO("/home/freystec/foosball-statistics/weights/yolov11l_imgsz_640.pt")  # load a pretrained model (recommended for training)

    # Train the model
    results = model.train(cfg="/home/freystec/foosball-statistics/configs/yolov11l_customized_augmentations_complete_labels.yaml")


if __name__ == '__main__':
    # multiprocessing.freeze_support()  # Uncomment if necessary
    main()



