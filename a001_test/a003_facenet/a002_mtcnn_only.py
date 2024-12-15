import os
import shutil

import cv2
import numpy as np
from facenet_pytorch.models.mtcnn import MTCNN
from matplotlib import pyplot as plt
from tqdm import tqdm

from a002_main.a001_utils.a002_general_utils import glob_png_paths_in_folder

IMAGE_FOLDER = "a002_main/a004_fastapi/a001_images/a003_upload_images_no_repeat"
SUCCESS_CROP_IMAGE_FOLDER = r"a001_test/a003_facenet/a001_mtcnn_success_crop"
FAIL_CROP_IMAGE_FOLDER = r"a001_test/a003_facenet/a002_mtcnn_fail_crop"
mtcnn = MTCNN(thresholds=[0.4, 0.5, 0.5])


def start():
    image_paths_list = glob_png_paths_in_folder(
        image_folder=IMAGE_FOLDER
    )

    fail_crop_image_paths_list = []
    success_crop_image_paths_list = []

    for image_path in tqdm(image_paths_list):
        image_array = cv2.imread(image_path)  # hwc, bgr, uint8
        image_array = cv2.cvtColor(image_array, cv2.COLOR_BGR2GRAY)
        image_array = image_array[:, :, None]
        image_array: np.ndarray
        image_array = np.tile(image_array, (1, 1, 3))

        rst_tensor = mtcnn(image_array)  # chw

        if rst_tensor is not None:
            rst_array = rst_tensor.numpy()
            rst_array = rst_array.transpose((1, 2, 0))  # hwc
            rst_array = (rst_array * 128 + 127.5).astype(np.uint8)

            success_crop_image_paths_list.append(image_path)

            # plt.imsave(
            #     fname=os.path.join(SUCCESS_CROP_IMAGE_FOLDER, os.path.basename(image_path)),
            #     arr=rst_array,
            # )

            # plt.imshow(rst_array)
            # plt.axis("off")
            # plt.show()
            # plt.close()

        else:
            fail_crop_image_paths_list.append(image_path)

            # shutil.copyfile(
            #     src=image_path,
            #     dst=os.path.join(FAIL_CROP_IMAGE_FOLDER, os.path.basename(image_path)),
            # )

    print(
        f"fails: {len(fail_crop_image_paths_list)} / {len(image_paths_list)}"
    )


if __name__ == '__main__':
    start()
