import json
import math
import os
from datetime import datetime
from glob import glob
from pathlib import Path
from typing import Any, Dict, List

import cv2
import natsort
import numpy as np
import torch
from colorama import Fore
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from torch.nn import functional

from a000_CONFIG import (
    DPI,
    TEST_NUM_SAMPLES_PER_EPOCH,
    DATASET_SF_TL54_PATH, LOGGER, FASTAPI_INFERENCE_WITH_MAX_IMAGE_SIDE,
)


# 已延迟导入 from a002_dataset_for_test import DatasetForTest


def my_collate_fn_factory(batch_size):
    if batch_size == 1:

        def my_collate_fn_for_batch_size_being_1(batch_dict_list):
            return batch_dict_list[0]

        return my_collate_fn_for_batch_size_being_1
    else:

        def my_collate_fn_for_batch_size_larger_than_1(batch_dict_list):
            return_dict: Dict[str, List[Any]] = {
                key: list() for key in batch_dict_list[0].keys()
            }
            for item_dict in batch_dict_list:
                for k, v in item_dict.items():
                    return_dict[k].append(v)
            return return_dict

        return my_collate_fn_for_batch_size_larger_than_1


def init_a_figure_and_an_axes():
    """
    处理figure上的axes。清空axes上的内容，关闭axis，位置设为占满figure。
    如果还没有axes，则创建一个满足要求的axes。
    """
    figure = plt.figure(
        num="my_figure",
        frameon=False,
        dpi=DPI,
        clear=True,
    )
    if not figure.get_axes():
        axes = figure.add_axes((0, 0, 1, 1))
    else:
        axes = figure.axes[0]
        axes.set_position((0, 0, 1, 1))
    axes.clear()
    axes.axis("off")
    return figure, axes


def adjust_figure_size_and_show_image_and_release_resources(
        img,
        figure: Figure,
        axes: Axes,
):
    """
    根据face尺寸调整figure尺寸。axes会自动随着figure调整位置，我们不用动axes。
    Args:
        img: HxWxC RGB
        figure:
        axes:
    Returns:
    """
    h, w, _ = img.shape  # HxWxC
    figure.set_size_inches(
        h=h / DPI,
        w=w / DPI,
    )

    axes.imshow(img)
    figure.show()

    plt.close(fig=figure)
    axes.clear()
    axes.axis("off")

    return figure


def save_to_json(obj, save_to_path):
    with open(save_to_path, "w") as file:
        json.dump(obj, file)


def load_json(load_from_path):
    with open(load_from_path, "r") as file:
        return json.load(file)


def build_dataset_for_test():
    from a002_model.a002_batch_test.a002_DatasetForTestOrVali import DatasetForTestOrVali

    return DatasetForTestOrVali(
        dataset_path=DATASET_SF_TL54_PATH,
        num_samples_per_epoch=TEST_NUM_SAMPLES_PER_EPOCH,
    )


def get_time_stamp_str():
    return datetime.now().strftime(format="%Y-%m-%d_%H-%M-%S")


def loss_penalty_func_for_d_an(x):
    return math.sin(math.pi / 4 * x + math.pi) + 1


def my_distance_func(tensor_0, tensor_1) -> torch.Tensor:
    """
    假设tensor的形状为 batch x feature_dim，返回tensor形状将是只有一维，长度为batch
    """
    return 1 - functional.cosine_similarity(tensor_0, tensor_1)


def glob_png_paths_in_folder(image_folder):
    image_path_list = glob(
        pathname=os.path.join(image_folder, "*.png"),
        recursive=False
    )
    return natsort.natsorted(image_path_list)


def read_image_path_as_hwc_bgr_uint8(image_path):
    image_array = cv2.imread(image_path)
    return image_array


def save_hwc_bgr_to_png(array, folder_path, filename):
    # array = cv2.cvtColor(src=array, code=cv2.COLOR_BGR2RGB)
    if not filename.lower().endswith(".png"):
        filename = f"{filename}.png"
    save_path = Path(folder_path) / Path(filename)
    cv2.imwrite(filename=str(save_path), img=array)
    # plt.imsave(save_path, array)
    LOGGER.info(
        Fore.GREEN +
        f"An image has been saved to {save_path.as_posix()}."
    )


def convert_hwc_bgr_uint8_to_gray_3c_uint8(arr):
    gray_array = cv2.cvtColor(arr, cv2.COLOR_BGR2GRAY)  # numpy hw uint8
    gray_array = np.expand_dims(gray_array, axis=2)  # numpy hwc c=1 uint8
    return np.tile(gray_array, (1, 1, 3))  # numpy hwc c=3 uint8


def ensure_max_side_for_hwc_bgr_uint8(
        img: np.ndarray,
        max_side: int = FASTAPI_INFERENCE_WITH_MAX_IMAGE_SIDE,
) -> np.ndarray:
    """
    确保输入图像的最长边不超过 max_side。
    如果超过，则按等比将最长边缩放至 max_side。

    参数:
    - img: 输入图像 (H, W, C)，BGR 格式，类型为 uint8。
    - max_side: 最长边限制，默认为 1000。

    返回:
    - 缩放后的图像 (H, W, C)，BGR，uint8。
    """
    if img is None:
        raise ValueError("输入图像为空（None）")

    h, w = img.shape[:2]
    longest_edge = max(h, w)

    # 判断是否需要缩放
    if longest_edge > max_side:
        # 计算缩放因子
        scale = max_side / float(longest_edge)
        new_w = int(w * scale)
        new_h = int(h * scale)
        # 使用 INTER_AREA 进行缩小，效果较好
        img = cv2.resize(
            img,
            (new_w, new_h),
            interpolation=cv2.INTER_AREA
        )

        LOGGER.info(
            Fore.GREEN +
            f"An image was resized, h w: ({h}, {w}) -> ({new_h}, {new_w})."
        )

    return img
