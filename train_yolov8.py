from ultralytics import YOLO


def main():
    model = YOLO(r"C:\Users\chris\Foosball Detector\runs\detect\train20\weights\best.pt")  # load a pretrained model (recommended for training)

    # Train the model
    results = model.train(
        data=r"D:\foosball_datasets\big_dataset_bratzdrum_freystein_bonn_combined\dataset.yaml", epochs=24,
        imgsz=1280, batch=4)


if __name__ == '__main__':
    # multiprocessing.freeze_support()  # Uncomment if necessary
    main()
