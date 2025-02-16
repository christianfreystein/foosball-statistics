from ultralytics import YOLO


def main():
    model = YOLO(r"C:\Users\chris\Foosball Detector\weights\yolov10m.pt")  # load a pretrained model (recommended for training)

    # Train the model
    results = model.train(
        data=r"D:\training_dataset_2\dataset.yaml", epochs=100,
        imgsz=640)


if __name__ == '__main__':
    # multiprocessing.freeze_support()  # Uncomment if necessary
    main()
