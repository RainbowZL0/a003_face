import traceback

try:
    # 你的代码
    1/0
except Exception as e:
    print(f"Error value: {e}")
    print(f"{traceback.format_exc()}")
