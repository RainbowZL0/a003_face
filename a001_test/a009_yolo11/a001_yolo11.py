import cv2
from matplotlib import pyplot as plt
from ultralytics import YOLO

from a002_model.a001_utils.a002_general_utils import convert_hwc_bgr_uint8_to_gray_3c_uint8, \
    ensure_max_side_for_hwc_bgr_uint8

MODEL_PATH = r"a001_test/a009_yolo11/models/yolov11s-face.pt"
OUTPUT_IMG_FOLDER = r"a001_test/a009_yolo11/output_images"

# FIXME not_face.png
INPUT_IMG_PATH_0 = r"a001_test/a009_yolo11/input_images/not_face.png"
INPUT_IMG_PATH_1 = r"a001_test/a009_yolo11/input_images/p2_0.png"


if __name__ == '__main__':
    model = YOLO(MODEL_PATH)  # load a custom model

    array_hwc_bgr_uint8_0 = cv2.imread(INPUT_IMG_PATH_0)
    array_hwc_bgr_uint8_1 = cv2.imread(INPUT_IMG_PATH_1)

    assert array_hwc_bgr_uint8_0 is not None and array_hwc_bgr_uint8_1 is not None

    array_list = [
        array_hwc_bgr_uint8_0,
        array_hwc_bgr_uint8_1,
    ]

    array_list = [convert_hwc_bgr_uint8_to_gray_3c_uint8(arr_i) for arr_i in array_list]

    # Predict with the model
    results = model.predict(
        array_list,
        save=True,
        project=OUTPUT_IMG_FOLDER,
        # device="cuda",
    )  # predict on an image

    resized_array_list = [ensure_max_side_for_hwc_bgr_uint8(arr_i) for arr_i in array_list]

    for i, rst in enumerate(results):
        if len(rst) == 0:  # 重要判断，不可忽略。原理是rst这个对象的内部有__len__魔术方法。
            print("没有检测到人脸")
        else:
            xyxy_n = rst.boxes.xyxyn
            x1_n, y1_n, x2_n, y2_n = xyxy_n[0, :].tolist()
            y_len, x_len = resized_array_list[i].shape[:2]
            xyxy_float = [x_len*x1_n, y_len*y1_n, x_len*x2_n, y_len*y2_n]
            x1, y1, x2, y2 = [int(i) for i in xyxy_float]
            face_array = resized_array_list[i][y1:y2, x1:x2]

            plt.axis('off')
            plt.imshow(cv2.cvtColor(face_array, cv2.COLOR_BGR2RGB))
            plt.show()
            plt.close()
    pass
