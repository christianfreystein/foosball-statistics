from ultralytics import YOLO


def main():
    model = YOLO('yolov8m.pt')  # load a pretrained model (recommended for training)

    # Train the model
    results = model.train(
        data=r"C:\Users\chris\foosball-dataset\big foosball dataset adjusted yolo format\dataset.yaml", epochs=100,
        imgsz=640)


if __name__ == '__main__':
    # multiprocessing.freeze_support()  # Uncomment if necessary
    main()
