# 这段代码包含多个可修复的问题
from typing import List, Optional
import sys, os  # 未排序的导入
import json


def process_data(data: List[str], flag=None):  # 缺少类型标注
    x = []  # 应使用更具体的类型标注
    for item in data:
        if flag == None:  # 应使用 is None
            continue
        x.append(item.lower)  # 缺少函数调用括号

    return x
