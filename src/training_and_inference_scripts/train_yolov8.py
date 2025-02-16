from ultralytics import YOLO


def main():
    model = YOLO(r"/runs/detect/train2/weights/best.pt")  # load a pretrained model (recommended for training)

    # Train the model
    results = model.train(
        data=r"D:\New_Big_Foosball_Dataset\Second_Prototype_Dataset_without_impossible\dataset.yaml", epochs=100,
        imgsz=640, batch=8)


if __name__ == '__main__':
    # multiprocessing.freeze_support()  # Uncomment if necessary
    main()
