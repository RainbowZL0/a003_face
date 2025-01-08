from ultralytics import YOLO

MODEL_PATH = r"a001_test/a009_yolo11/models/yolov11s-face.pt"
INPUT_IMG_PATH = r"a001_test/a009_yolo11/input_images/p2_0.png"


if __name__ == '__main__':
    # Load a model
    # model = YOLO("yolo11n.pt")  # load an official model
    model = YOLO(MODEL_PATH)  # load a custom model

    # Predict with the model
    results = model.predict(INPUT_IMG_PATH, save=True)  # predict on an image
    print(results)

    boxes = results[0].boxes
    print(boxes)
