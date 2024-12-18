import hashlib
from pathlib import Path

import cv2
from tqdm import tqdm

from a002_model.a001_utils.a002_general_utils import glob_png_paths_in_folder

IMAGE_FOLDER = r"a003_fastapi/a001_images/a003_upload_images_no_repeat"


def start():
    image_path_list = glob_png_paths_in_folder(IMAGE_FOLDER)

    mismatched_image_path_list = []
    sha256_code_list = []

    for image_path in tqdm(image_path_list):
        # path形式类似 2024-12-17_16-02-33_76bdc3092f56f3ed3e7bfd6530344be4.png
        file_name = Path(image_path).stem
        md5_code_from_filename = file_name.split("_")[2]

        array = cv2.imread(image_path)
        if array is None:
            print(f"Read image failed at {image_path}.")
            continue
        arr_bytes = array.tobytes()

        # md5_code = hashlib.md5(arr_bytes).hexdigest()
        # if md5_code_from_filename != md5_code:
        #     print(f"md5 code mismatched at {image_path}.")
        #     mismatched_image_path_list.append(image_path)

        sha256_code = hashlib.sha256(arr_bytes).hexdigest()
        sha256_code_list.append(sha256_code)

    # print(f"Number of mismatched images: {len(mismatched_image_path_list)} / {len(image_path_list)}")
    no_repeat_sha256_code_list = list(set(sha256_code_list))
    print(len(no_repeat_sha256_code_list))
    print(len(sha256_code_list))


if __name__ == '__main__':
    start()
