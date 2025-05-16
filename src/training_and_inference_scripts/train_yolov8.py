from ultralytics import YOLO


def main():
    model = YOLO("/home/freystec/foosball-statistics/weights/yolo11m.pt")  # load a pretrained model (recommended for training)

    # Train the model
    results = model.train(
        data="/home/freystec/Foosball_Datasets/Second_Prototype_Dataset_with_Leonhart_Topview_Data/dataset.yaml", epochs=50,
        imgsz=1280, batch=4)


if __name__ == '__main__':
    # multiprocessing.freeze_support()  # Uncomment if necessary
    main()



