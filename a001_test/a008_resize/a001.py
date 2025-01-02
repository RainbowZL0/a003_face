import cv2
import numpy as np
from colorama import Fore

from a002_model.a001_utils.a000_CONFIG import LOGGER
from a002_model.a001_utils.a002_general_utils import save_hwc_bgr_to_png

INPUT_IMG_PATH = r"a001_test/a008_resize/input_images/p2_0.png"
OUTPUT_IMG_FOLDER = r"a001_test/a008_resize/output_images"


def ensure_max_side(img: np.ndarray, max_side: int = 1000) -> np.ndarray:
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
        img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

        LOGGER.info(
            Fore.GREEN +
            f"An image was resized, h w: ({h}, {w}) -> ({new_h}, {new_w})."
        )

    return img


def start():
    # 读取一张大分辨率图片（示例）
    img_path = INPUT_IMG_PATH
    img = cv2.imread(img_path, flags=cv2.IMREAD_COLOR)  # 读入 BGR, uint8

    # 确保图像最长边 <= 1000
    img_resized = ensure_max_side(img, max_side=1000)
    save_hwc_bgr_to_png(img_resized, OUTPUT_IMG_FOLDER, "haha")

    # 显示结果
    print(f"原图尺寸: {img.shape}")
    print(f"缩放后尺寸: {img_resized.shape}")

    # 继续你的后续操作（人脸检测、保存、展示等）
    # ...


if __name__ == "__main__":
    start()
