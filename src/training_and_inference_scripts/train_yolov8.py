from ultralytics import YOLO


def main():
    model = YOLO(r"D:\Foosball Detector\runs\detect\train23\weights\best_imgsz_1280.pt")  # load a pretrained model (recommended for training)

    # Train the model
    results = model.train(
        data=r"D:\New_Big_Foosball_Dataset\Second_Prototype_Dataset_with_Leonhart_Topview_Data\dataset.yaml", epochs=50,
        imgsz=1280, batch=4)


if __name__ == '__main__':
    # multiprocessing.freeze_support()  # Uncomment if necessary
    main()



