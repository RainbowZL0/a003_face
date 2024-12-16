import cv2
import numpy as np
from insightface.app import FaceAnalysis
from matplotlib import pyplot as plt

from a001_test.a001_opencv_detection.a001_bug_check import (
    read_image_path_as_hwc_bgr_uint8,
    glob_png_paths_in_folder,
)

IMAGE_FOLDER = r"a002_main/a004_fastapi/a001_images/a003_upload_images_no_repeat"


def start():
    # 初始化人脸分析器
    app = FaceAnalysis(providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
    # 或仅用CPU：app = FaceAnalysis(providers=['CPUExecutionProvider'])

    # 初始化模型
    app.prepare(ctx_id=0, det_size=(640, 640))

    image_paths_list = glob_png_paths_in_folder(
        image_folder=IMAGE_FOLDER
    )

    for image_path in image_paths_list:
        # 读取图片
        img = read_image_path_as_hwc_bgr_uint8(image_path)

        # 检测人脸
        faces = app.get(img)

        # 处理检测结果
        for face in faces:
            # 获取人脸框
            bbox = face.bbox.astype(int)
            # 获取关键点
            landmarks = face.kps.astype(int)

            # 绘制人脸框
            cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)

            # 绘制关键点
            for landmark in landmarks:
                cv2.circle(img, (landmark[0], landmark[1]), 2, (0, 255, 0), -1)

            plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            plt.axis('off')
            plt.show()
            plt.close()


def align_face(image, landmarks, desired_size=112):
    # 使用5点关键点进行对齐
    src = np.array([
        [30.2946, 51.6963],  # 左眼中心
        [65.5318, 51.6963],  # 右眼中心
        [48.0252, 71.7366],  # 鼻尖
        [33.5493, 92.3655],  # 左嘴角
        [62.7299, 92.3655]],  # 右嘴角
        dtype=np.float32)

    # 根据输出尺寸缩放标准点
    src = src * desired_size / 112

    # 计算相似变换矩阵
    tform = cv2.estimateAffinePartial2D(landmarks, src)[0]

    # 应用变换
    aligned_face = cv2.warpAffine(image, tform, (desired_size, desired_size))

    return aligned_face


# 使用示例:
# image 是原始图像
# face_info 是insightface返回的信息
def get_aligned_face(image, face_info, size=112):
    # 获取人脸框
    bbox = face_info.bbox.astype(np.int32)
    # 获取关键点
    landmarks = face_info.kps

    # 截取人脸区域（可选，也可以直接对整张图做变换）
    face = image[bbox[1]:bbox[3], bbox[0]:bbox[2]]

    # 对齐人脸
    aligned_face = align_face(image, landmarks, size)

    return aligned_face


if __name__ == '__main__':
    start()
