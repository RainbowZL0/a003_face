import cv2
import numpy as np
import torch
from retinaface.pre_trained_models import get_model
from tqdm import tqdm

from a002_model.a001_utils.a002_general_utils import glob_png_paths_in_folder

IMAGE_FOLDER = "a003_fastapi/a001_images/a003_upload_images_no_repeat"
CROP_IMAGE_FOLDER = r"a001_test/a003_facenet/a001_mtcnn_success_crop"

model = get_model("resnet50_2020-07-20", max_size=2048)
model.eval()


def start():
    image_paths_list = glob_png_paths_in_folder(
        image_folder=IMAGE_FOLDER
    )
    for image_path in tqdm(image_paths_list):
        image_array = cv2.imread(image_path)  # hwc, bgr, uint8
        image_array = cv2.cvtColor(image_array, cv2.COLOR_BGR2GRAY)
        image_array = image_array[:, :, None]  # hwc, gray, channels=1, uint8
        image_array: np.ndarray
        image_array = np.tile(image_array, (1, 1, 3))  # hwc, gray, channels=3, uint8

        annotation = model.predict_jsons(image_array)
        pass


if __name__ == '__main__':
    start()
