import os
from glob import glob
from pathlib import Path
from natsort import natsorted

import cv2

IMG_FOLDER = './a001_opencv_detection/imgs'


def main():
    # 加载OpenCV的预训练人脸检测模型
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )

    img_names = glob(pathname='*.png', root_dir=IMG_FOLDER, recursive=False)
    img_names = natsorted(img_names)

    for image_name in img_names:
        image_path = os.path.join(IMG_FOLDER, image_name)
        image = cv2.imread(image_path)

        # 将图片转换为灰度图（因为人脸检测一般在灰度图上进行）
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # 使用人脸检测器检测图片中的人脸
        faces = face_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=8,
        )

        # 如果检测到人脸，框住并裁剪
        for x, y, w, h in faces:
            # 绘制矩形框在原图上
            cv2.rectangle(
                img=image,
                pt1=(x, y),
                pt2=(x + w, y + h),
                color=(0, 255, 0),
                thickness=2,
            )

            # # 裁剪出人脸区域
            # cropped_face = image[y: y + h, x: x + w]
            # # 保存裁剪的人脸到新文件
            # cv2.imwrite(
            #     "a001_opencv_detection/cropped_face.jpg",
            #     cropped_face,
            # )

        original_img_name = Path(image_path).stem
        # 保存带有人脸框的图片
        cv2.imwrite(
            f"./a001_opencv_detection/faces/{original_img_name}.png",
            image,
        )

    # # 显示结果（可选）
    # cv2.imshow("Detected Faces", image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()


if __name__ == '__main__':
    pass
    main()
