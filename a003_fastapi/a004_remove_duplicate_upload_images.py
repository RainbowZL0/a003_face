import os
import shutil
import time
from collections import defaultdict
from glob import glob
from pathlib import Path

import schedule

from a003_fastapi.a002_my_fastapi_processor import get_time_stamp_str

# 使用 glob 获取所有图片文件路径
SOURCE_DIR = r"a003_fastapi/a001_images/a001_upload_images"
MOVE_TO_DIR = (
    r"a003_fastapi/a001_images/a003_upload_images_no_repeat"  # 目标文件夹
)

# 确保目标文件夹存在
os.makedirs(MOVE_TO_DIR, exist_ok=True)


def get_group_dict_with_hash_as_key_from_image_dir(source_dir):
    image_paths = glob(os.path.join(source_dir, "*.png"))

    # 用hash值分组
    hash_groups = defaultdict(list)
    for image_path in image_paths:
        # glob返回完整路径，所以需要用basename获取文件名
        filename = Path(image_path).stem
        try:
            hash_value = filename.split("_")[2]
        except Exception as e:
            print(f"At {image_path}, an error occurred: {e}")
            exit(1)
        else:
            hash_groups[hash_value].append(image_path)
    return hash_groups


def remove_duplicated(source_dir, whether_move_to_new_dir, move_to_dir):
    if whether_move_to_new_dir and move_to_dir is None:
        raise TypeError(
            "你选择了whether_move_dir_new_dir=True，然而没有指定"
            "move_to_dir目标位置，现在它取值None。"
        )
    group_dict_with_hash_as_key = get_group_dict_with_hash_as_key_from_image_dir(
        source_dir
    )
    # 对每组重复图片，移动一个到新文件夹，删除其他
    for hash_value, files in group_dict_with_hash_as_key.items():
        if len(files) >= 1:
            # 删除其余重复文件
            for f in files[1:]:
                os.remove(f)
                print(f"Removed duplicate: {os.path.basename(f)}")
            # 移动第一个文件到新文件夹
            if whether_move_to_new_dir:
                source_file = files[0]
                # using str() to suppress ide warning
                target_file = str(
                    os.path.join(move_to_dir, os.path.basename(source_file))
                )
                shutil.move(source_file, target_file)
                print(f"Moved to {move_to_dir}: {os.path.basename(source_file)}")


def calculate_png_num_in_folder(folder_path):
    lst = glob(pathname="*.png", root_dir=folder_path)
    return len(lst)


def start():
    remove_duplicated(
        source_dir=SOURCE_DIR,
        whether_move_to_new_dir=True,
        move_to_dir=MOVE_TO_DIR,
    )
    remove_duplicated(
        source_dir=MOVE_TO_DIR,
        whether_move_to_new_dir=False,
        move_to_dir=None,
    )
    print(
        f"\n"
        f"Done.\n"
        f"Num of pngs in {SOURCE_DIR} = {calculate_png_num_in_folder(SOURCE_DIR)}.\n"
        f"Num of pngs in {MOVE_TO_DIR} = {calculate_png_num_in_folder(MOVE_TO_DIR)}.\n"
        f"Current time: {get_time_stamp_str()}"
    )


def start_every_given_seconds(interval=3600):
    schedule.every(interval).seconds.do(start)
    start()
    while True:
        schedule.run_pending()
        time.sleep(1)


if __name__ == "__main__":
    start_every_given_seconds(500)
