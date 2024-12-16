import os
import shutil

import cv2
import numpy as np
from deepface import DeepFace
from tqdm import tqdm

from a002_main.a001_utils.a002_general_utils import glob_png_paths_in_folder

IMAGE_FOLDER = "a002_main/a004_fastapi/a001_images/a003_upload_images_no_repeat"
SUCCESS_CROP_IMAGE_FOLDER = r"a001_test/a006_retina_face_test/a001_retinaface_success_crop"
FAIL_CROP_IMAGE_FOLDER = r"a001_test/a006_retina_face_test/a002_retinaface_fail_crop"

SAVE_RESULT = False


def start():
    image_paths_list = glob_png_paths_in_folder(
        image_folder=IMAGE_FOLDER
    )

    fail_crop_image_paths_list = []
    success_crop_image_paths_list = []

    for image_path in tqdm(image_paths_list):
        try:
            face_dict_list = DeepFace.extract_faces(
                img_path=cv2.imread(image_path),
                detector_backend="retinaface",
                enforce_detection=False,
            )
        except Exception as e:
            print(
                f"在该图片上没有发现人脸：{image_path}.\n"
                f"Exception: {e}"
            )
            fail_crop_image_paths_list.append(
                os.path.basename(image_path)
            )

            if SAVE_RESULT:
                shutil.copyfile(
                    src=image_path,
                    dst=os.path.join(
                        FAIL_CROP_IMAGE_FOLDER,
                        os.path.basename(image_path),
                    ),
                )
        else:
            # TODO 如果没有检测到人脸，则confidence项会等于0，应当改为据此判断是否检测到了人脸；现在是用exception
            # face_dict_list[0]: dict {
            #   "face": ndarray,  # 注意格式是 RGB float HWC 0~1
            #   "facial_area": {},
            #   "confidence": float,
            # }
            success_crop_image_paths_list.append(image_path)

            the_first_face = face_dict_list[0]["face"]
            the_first_face *= 255
            the_first_face = the_first_face.astype(np.uint8)  # to hwc rgb uint8
            the_first_face = cv2.cvtColor(the_first_face, cv2.COLOR_RGB2BGR)  # to hwc bgr uint8

            if SAVE_RESULT:
                cv2.imwrite(
                    filename=os.path.join(
                        SUCCESS_CROP_IMAGE_FOLDER,
                        os.path.basename(image_path)
                    ),
                    img=the_first_face,
                )

    print(
        f"failed: {len(fail_crop_image_paths_list)} / {len(image_paths_list)}.\n"
        f"success: {len(success_crop_image_paths_list)} / {len(image_paths_list)}."
    )


if __name__ == '__main__':
    start()
