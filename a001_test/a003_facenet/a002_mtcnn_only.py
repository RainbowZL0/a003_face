import cv2
import numpy as np
from facenet_pytorch.models.mtcnn import MTCNN
from matplotlib import pyplot as plt
from tqdm import tqdm

from a002_main.a001_utils.a002_general_utils import glob_png_paths_in_folder

IMAGE_FOLDER = "a002_main/a004_fastapi/a001_images/a003_upload_images_no_repeat"
CROP_IMAGE_FOLDER = r"a001_test/a003_facenet/a001_mtcnn_only"
mtcnn = MTCNN()

def start():
    image_paths_list = glob_png_paths_in_folder(
        image_folder=IMAGE_FOLDER
    )

    fail_count = 0

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

            plt.imshow(rst_array)
            plt.axis("off")
            plt.show()
            plt.close()

        else:
            fail_count += 1

    print(
        f"fails: {fail_count} / {len(image_paths_list)}"
    )


if __name__ == '__main__':
    start()
