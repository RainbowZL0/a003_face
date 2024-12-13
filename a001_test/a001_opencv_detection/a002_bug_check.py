import os.path
from tqdm import tqdm

import cv2
from glob import glob

from a002_main.a001_utils.a002_general_utils import (
    glob_png_paths_in_folder,
    read_image_path_as_hwc_bgr_uint8,
)

IMAGE_FOLDER = r"a002_main/a004_fastapi/a001_images/a003_upload_images_no_repeat"





def start():
    opencv_face_detector = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )

    image_paht_list = glob_png_paths_in_folder(IMAGE_FOLDER)
    for image_path in tqdm(image_paht_list):
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
