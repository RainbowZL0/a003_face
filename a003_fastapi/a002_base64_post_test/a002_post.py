import json
import time

import colored_traceback
import requests

from a002_model.a001_utils.a002_general_utils import get_time_stamp_str

colored_traceback.add_hook()

POST_REPETITION_TIMES = 1000
POST_TO_PORT = 8092

API_URL = rf"http://8.130.179.197:{POST_TO_PORT}"
# API_URL = rf"http://8.141.155.10:{POST_TO_PORT}"
# API_URL = rf"http://127.0.0.1:{POST_TO_PORT}"
# API_URL = rf"http://172.29.248.181:{POST_TO_PORT}"

BASE64_TXT_PATH_0 = r"a003_fastapi/a002_base64_post_test/base64/p1_0.txt"
BASE64_TXT_PATH_1 = r"a003_fastapi/a002_base64_post_test/base64/p2_0.txt"


def read_base64_from_file(file_path: str) -> str:
    """从文件读取base64字符串"""
    with open(file_path, 'r') as f:
        return f.read().strip()


def post_image_pair_base64(
        base64_str_0: str,
        base64_str_1: str,
        api_url: str
) -> dict:
    endpoint = f"{api_url}/facecomparsion"

    json_data = {
        "image_0": base64_str_0,
        "image_1": base64_str_1
    }

    response = requests.post(
        url=endpoint,
        json=json_data,
        headers={"accept": "application/json"}
    )

    response.raise_for_status()
    return response.json()


def high_level_api_for_post_image_pair_base64(base64_str_0, base64_str_1):
    result = post_image_pair_base64(
        base64_str_0=base64_str_0,
        base64_str_1=base64_str_1,
        api_url=API_URL
    )
    result_str = json.dumps(result, indent=4, ensure_ascii=False)
    print(result_str)


def print_post_format(base64_path_0: str, base64_path_1: str, api_url: str):
    base64_image_0 = read_base64_from_file(base64_path_0)
    base64_image_1 = read_base64_from_file(base64_path_1)

    endpoint = f"{api_url}/facecomparsion"
    json_data = {
        "image_0": base64_image_0,
        "image_1": base64_image_1
    }

    # 准备请求但不发送
    req = requests.Request(
        'POST',
        endpoint,
        json=json_data,
        headers={"accept": "application/json"}
    )
    prepared = req.prepare()
    pass


def repeat_post_test():
    base64_str_0 = read_base64_from_file(BASE64_TXT_PATH_0)
    base64_str_1 = read_base64_from_file(BASE64_TXT_PATH_1)

    starting_time = time.time()

    for i in range(POST_REPETITION_TIMES):
        high_level_api_for_post_image_pair_base64(
            base64_str_0=base64_str_0,
            base64_str_1=base64_str_1,
        )

    ending_time = time.time()

    time_elapsed = round(ending_time - starting_time, 3)

    print(
        f"Posted {POST_REPETITION_TIMES} times in {time_elapsed} seconds. \n"
        f"Average cost = {time_elapsed / POST_REPETITION_TIMES} seconds."
    )


if __name__ == "__main__":
    repeat_post_test()
