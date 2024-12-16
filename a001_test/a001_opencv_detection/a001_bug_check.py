import cv2
from tqdm import tqdm

from a002_main.a001_utils.a002_general_utils import (
    glob_png_paths_in_folder,
    read_image_path_as_hwc_bgr_uint8,
)

IMAGE_FOLDER = r"a001_test/a001_opencv_detection/imgs"


def start():
    opencv_face_detector = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )

    image_path_list = glob_png_paths_in_folder(IMAGE_FOLDER)
    for image_path in tqdm(image_path_list):
        image_array = read_image_path_as_hwc_bgr_uint8(image_path)
        image_array = cv2.cvtColor(image_array, cv2.COLOR_BGR2GRAY)
        try:
            faces = opencv_face_detector.detectMultiScale(
                image_array,
                scaleFactor=1.1,
                minNeighbors=8,
            )
        except Exception as e:
            print(e)
            print(f"error occurred at {image_path}")


if __name__ == '__main__':
    start()
    pass
