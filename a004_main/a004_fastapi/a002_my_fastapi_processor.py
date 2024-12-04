import base64
import datetime
import hashlib
import json
import pprint
from io import BytesIO
from pathlib import Path
from typing import no_type_check

import cv2
import numpy
import numpy as np
import torch
from PIL import Image
from colorama import Fore
from facenet_pytorch.models.inception_resnet_v1 import InceptionResnetV1
from fastapi import UploadFile, File
from matplotlib import pyplot as plt
from torchvision.transforms import v2

from a004_main.a001_utils.a000_CONFIG import (
    FASTAPI_LOG_JSON_FILE_PATH,
    FASTAPI_UPLOAD_IMAGE_FOLDER,
    LOGGER,
    LOAD_FROM_STATE_PATH,
    TRAINING_OR_VALI_DEVICE,
    DISTANCE_THRESHOLD,
    FASTAPI_CROP_IMAGE_FOLDER,
    FASTAPI_DEVICE,
    FASTAPI_USING_DETECTION_METHOD,
)
from a004_main.a001_utils.a002_general_utils import my_distance_func


class MyFastapiProcessor:
    def __init__(self):
        self.model = build_model_and_load_my_state_for_fastapi()
        self.transform = v2.Compose([
            v2.ToImage(),  # ndarray HWC -> tensor CHW, dtype不变
            v2.ToDtype(torch.float32),
            v2.Resize((160, 160)),
            v2.Normalize(mean=(127.5, 127.5, 127.5), std=(128, 128, 128)),
        ])

        if FASTAPI_USING_DETECTION_METHOD == "deepface":
            from deepface.DeepFace import extract_faces

            self.deepface_extract_faces = extract_faces
        elif FASTAPI_USING_DETECTION_METHOD == "opencv":
            self.opencv_face_detector = cv2.CascadeClassifier(
                cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            )

    def get_image_pair_and_verify_file_version(
            self,
            file_0: UploadFile = File(...),
            file_1: UploadFile = File(...),
    ):
        """
        读取图片的流程
            1. 写入到磁盘。
                file类型 -> file.read()得到二进制数据 -> file.seek(0)重置读取指针至文件开头 -> 'rb'模式写入到磁盘
            2. file类型 读取为tensor
                file类型 -> PIL.open() -> numpy.array() -> 通道顺序处理，uin8处理，归一化处理 -> tensor
            3. base64字符串 读取为tensor
                decode为二进制 -> ByteIO()包装 -> Image.open() -> numpy.array()
        """
        filename_list = [f_i.filename for f_i in [file_0, file_1]]

        for f_i in [file_0, file_1]:
            save_upload_file_obj_to_disk_as_image(f_i)

        img_arr_list = [
            read_upload_file_img_as_numpy_hwc_bgr_uint8(f_i) for f_i in [file_0, file_1]
        ]

        # face in face_array_list is HWC, BGR, uint8
        face_array_list = [
            self.crop_face_from_img_to_hwc_bgr_uint8(arr_i) for arr_i in img_arr_list
        ]

        # 保存crop图片看看人脸位置是否准确
        for i in range(2):
            save_hwc_bgr_to_png(
                array=face_array_list[i],
                folder_path=FASTAPI_CROP_IMAGE_FOLDER,
                filename=filename_list[i],
            )

        # 转为tensor, 1CHW, RGB, float, -1 ~ 1
        face_tensor_list = [
            self.transform_from_array_to_tensor(face_array)
            for face_array in face_array_list
        ]
        distance = self.infer_distance_given_face_tensor_list(face_tensor_list)

        result_dict = {
            "distance": distance,
            "is_same_person": judge_using_distance_threshold(distance),
        }

        # 输出处理完成的日志
        LOGGER.info(
            Fore.LIGHTGREEN_EX
            + f"Image pair '{filename_list[0]}', '{filename_list[1]}' done.\n"
              f"{pprint.pformat(result_dict)}"
        )

        return result_dict

    def get_image_pair_and_verify_base64_version(
            self,
            image_0: str,
            image_1: str,
    ):
        """
        解码过程为，base64 -> decode -> ByteIO() -> PIL.open() -> numpy.array()。
        UploadFile有文件名，而base64没有，不便于保存到本地。采用时间戳生成文件名。
        """
        img_arr_list = [
            read_base64_as_np_hwc_bgr_uint8(code_i) for code_i in [image_0, image_1]
        ]

        filename_tuple = generate_a_pair_of_file_name_from_array(
            img_arr_list[0], img_arr_list[1]
        )
        for i in range(2):
            save_hwc_bgr_to_png(
                array=img_arr_list[i],
                folder_path=FASTAPI_UPLOAD_IMAGE_FOLDER,
                filename=filename_tuple[i],
            )

        face_arr_list = [
            self.crop_face_from_img_to_hwc_bgr_uint8(img_i) for img_i in img_arr_list
        ]
        for i in range(2):
            save_hwc_bgr_to_png(
                array=face_arr_list[i],
                folder_path=FASTAPI_CROP_IMAGE_FOLDER,
                filename=filename_tuple[i],
            )

        face_tensor_list = [
            self.transform_from_array_to_tensor(arr_i) for arr_i in face_arr_list
        ]
        distance = self.infer_distance_given_face_tensor_list(face_tensor_list)

        result_dict = {
            "error_code": 0,
            "error_message": "SUCCESS",
            "timestamp": get_time_stamp_str(),
            "result": {
                "score": transform_distance_to_similarity_score(distance),
                "is_the_same_person": judge_using_distance_threshold(distance),
            },
        }

        log_dict = {
            **result_dict,
            "log": {
                "filename_image_0": filename_tuple[0],
                "filename_image_1": filename_tuple[1],
            }
        }
        save_log_by_appending_to_json(
            log_dict=log_dict,
            log_json_file_path=FASTAPI_LOG_JSON_FILE_PATH,
        )

        # 输出处理完成的日志
        LOGGER.info(
            Fore.LIGHTGREEN_EX + f"Image pair done.\n{pprint.pformat(log_dict)}"
        )

        return result_dict

    def transform_from_array_to_tensor(self, face_array):
        face_array = cv2.cvtColor(src=face_array, code=cv2.COLOR_BGR2RGB)
        face_tensor = self.transform(face_array)
        return face_tensor.to(device=FASTAPI_DEVICE).unsqueeze(0)

    def infer_distance_given_face_tensor_list(self, face_tensor_list):
        """推理"""
        self.model.eval()
        with torch.no_grad():
            out_0, out_1 = (self.model(face_array) for face_array in face_tensor_list)
            distance = my_distance_func(tensor_0=out_0, tensor_1=out_1).item()
        return round(distance, 5)

    def crop_face_from_img_to_hwc_bgr_uint8(self, arr: np.ndarray):
        """
        arr: np_hwc_bgr_uint8
        return:
            np_hwc_bgr_uint8
        """
        if FASTAPI_USING_DETECTION_METHOD == "deepface":
            try:
                face_dict_list = self.deepface_extract_faces(
                    img_path=arr,
                    detector_backend="retinaface",
                )
            except Exception as e:
                LOGGER.error(msg=f"{e}")
            else:
                # face: dict {
                #   "face": ndarray,  # 注意格式是 BGR float HxWxC
                #   "facial_area": {},
                #   "confidence": float,
                # }
                face_array: np.ndarray
                face_array = face_dict_list[0]["face"]
                face_array *= 255
                return face_array.astype(np.uint8)
        elif FASTAPI_USING_DETECTION_METHOD == "opencv":
            gray = cv2.cvtColor(arr, cv2.COLOR_BGR2GRAY)
            faces = self.opencv_face_detector.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=8,
            )
            # 如果没有检测到人脸，return 原图像
            if isinstance(faces, tuple):
                LOGGER.info(
                    Fore.GREEN + "No face was detected, return the original image."
                )
                return arr
            else:
                (x, y, w, h) = faces[0]
                face_array = arr[y: y + h, x: x + w, :]
                plt.imshow(cv2.cvtColor(face_array, cv2.COLOR_BGR2RGB))
                plt.show()
                return face_array
        else:
            raise ValueError(
                f"参数FASTAPI_USING_DETECTION_METHOD的取值不受支持。"
                f"当前取值为{FASTAPI_USING_DETECTION_METHOD}，"
                f"支持的取值为'opencv'或'deepface'。"
            )


def read_base64_as_np_hwc_bgr_uint8(code_0: str):
    byte = base64.b64decode(code_0)
    arr = numpy.array(Image.open(BytesIO(byte)))
    arr = arr.astype(np.uint8)

    if arr.shape[2] == 3:  # RGB and RGBA have different operations
        color_convert_code = cv2.COLOR_RGB2BGR
    else:
        color_convert_code = cv2.COLOR_RGBA2BGR
    return cv2.cvtColor(arr, color_convert_code)


def transform_distance_to_similarity_score(distance):
    """0 ~ 2 -> 0 ~ 100"""
    sim = (-distance) * 50 + 100
    return round(sim, 5)


@no_type_check
def read_upload_file_img_as_numpy_hwc_bgr_uint8(file_0: UploadFile) -> np.ndarray:
    """
    Returns: HWC, RGB, uint8
    """
    contents = file_0.file
    img = Image.open(contents)
    img = np.array(img)
    if img.shape[2] == 3:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    else:
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
    img = img.astype(np.uint8)
    return img


def save_upload_file_obj_to_disk_as_image(f_0: UploadFile = File(...)):
    # 保存图片
    if f_0.filename is not None:
        filename: str = f_0.filename
    else:
        raise ValueError(
            "filename should not be None!"
        )
    file_obj = f_0.file

    upload_image_folder = Path(FASTAPI_UPLOAD_IMAGE_FOLDER)
    if not upload_image_folder.exists():
        upload_image_folder.mkdir(parents=True, exist_ok=True)
    upload_image_path = upload_image_folder / Path(filename)
    with open(upload_image_path, "wb") as f:
        content = file_obj.read()
        file_obj.seek(0)
        f.write(content)


def save_hwc_bgr_to_png(array, folder_path, filename):
    array = cv2.cvtColor(src=array, code=cv2.COLOR_BGR2RGB)
    if not filename.lower().endswith(".png"):
        filename = f"{filename}.png"
    save_path = Path(folder_path) / Path(filename)
    # cv2.imwrite(filename=str(save_path), img=array)
    plt.imsave(save_path, array)


def judge_using_distance_threshold(distance):
    if distance <= DISTANCE_THRESHOLD:
        return True
    else:
        return False


def build_model_and_load_my_state_for_fastapi():
    """
    存储时的格式如下，
    state = {
        "model_state": self.model.state_dict(),
        "optimizer_state": self.optimizer.state_dict(),
        "scheduler_state": self.scheduler.state_dict(),
        "current_epochs": self.current_epochs + 1,
        "current_iters_in_an_epoch": self.current_iters_in_an_epoch,
        "iters_up_to_now": self.iters_up_to_now,
    }
    """
    LOGGER.info(
        Fore.LIGHTGREEN_EX
        + f"Building model and loading state for FastAPI, from {LOAD_FROM_STATE_PATH}."
    )
    read_state = torch.load(LOAD_FROM_STATE_PATH, map_location=TRAINING_OR_VALI_DEVICE)

    model = InceptionResnetV1(pretrained="vggface2").to(device=FASTAPI_DEVICE)
    model.load_state_dict(state_dict=read_state["model_state"])
    return model


def get_time_stamp_str():
    return datetime.datetime.now().strftime(format="%Y-%m-%d_%H-%M-%S")


def generate_a_pair_of_file_name_from_array(arr_0, arr_1):
    time_str = get_time_stamp_str()
    arr_0_file_name = generate_a_file_name_from_array(time_str, arr_0)
    arr_1_file_name = generate_a_file_name_from_array(time_str, arr_1)
    return arr_0_file_name, arr_1_file_name


def generate_a_file_name_from_array(time_str, arr):
    arr_bytes = arr.tobytes()
    md5_hash = hashlib.md5(arr_bytes).hexdigest()

    return f"{time_str}_{md5_hash}.png"


def save_log_by_appending_to_json(log_dict, log_json_file_path):
    with open(log_json_file_path, "a") as file:
        file.write(json.dumps(log_dict) + "\n")
