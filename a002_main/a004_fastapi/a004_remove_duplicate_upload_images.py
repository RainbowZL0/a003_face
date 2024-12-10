import os
import glob
import shutil
from collections import defaultdict
from pathlib import Path

# 使用 glob 获取所有图片文件路径
image_dir = r'a002_main/a004_fastapi/a001_images/a001_upload_images'
move_to_dir = r'a002_main/a004_fastapi/a001_images/a003_upload_images_no_repeat'  # 目标文件夹

# 确保目标文件夹存在
os.makedirs(move_to_dir, exist_ok=True)

image_paths = glob.glob(os.path.join(image_dir, '*.png'))

# 用hash值分组
hash_groups = defaultdict(list)
for img in image_paths:
    # glob返回完整路径，所以需要用basename获取文件名
    filename = Path(img).stem
    hash_value = filename.split('_')[2]
    hash_groups[hash_value].append(img)

# 对每组重复图片，移动一个到新文件夹，删除其他
for hash_value, files in hash_groups.items():
    if len(files) >= 1:
        # 移动第一个文件到新文件夹
        source_file = files[0]
        target_file = os.path.join(move_to_dir, os.path.basename(source_file))
        shutil.move(source_file, target_file)
        print(f'Moved to {move_to_dir}: {os.path.basename(source_file)}')
        # 删除其余重复文件
        for f in files[1:]:
            os.remove(f)
            print(f'Removed duplicate: {os.path.basename(f)}')
