import base64
import os
from glob import glob
from pathlib import Path

IMG_FOLDER_PATH = r'./a002_main/a004_fastapi/a002_base64_post_test/images'
BASE64_FOLDER_PATH = r'./a002_main/a004_fastapi/a002_base64_post_test/base64'


def image_to_base64_file(image_path):
    # 检查文件是否存在
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"找不到图片文件: {image_path}")

    # 检查文件是否是PNG格式
    if not image_path.lower().endswith('.png'):
        raise ValueError("文件必须是PNG格式")

    try:
        # 读取图片文件
        with open(image_path, 'rb') as image_file:
            # 将图片转换为base64编码
            encoded_string = base64.b64encode(image_file.read())
            image_file.seek(0)

        # 生成输出文件名（将.png替换为.txt）
        output_path = Path(BASE64_FOLDER_PATH) / Path(Path(image_path).stem + '.txt')

        # 将base64编码写入文本文件
        with open(output_path, 'wb') as text_file:
            text_file.write(encoded_string)

        print(f"转换完成! 已保存到: {output_path}")

    except Exception as e:
        print(f"转换过程中出现错误: {str(e)}")


def img_folder_to_base64(image_folder_path):
    image_file_names = glob(
        pathname="*.png",
        root_dir=IMG_FOLDER_PATH,
        recursive=False,
    )
    for image_file_name in image_file_names:
        image_path = os.path.join(image_folder_path, image_file_name)
        image_to_base64_file(image_path)


if __name__ == "__main__":
    img_folder_to_base64(IMG_FOLDER_PATH)
    pass
