import os
import shutil

file_path = r"a001_test/a005_debugpy_test/test.txt"


def start():
    with open(file_path, 'w') as file:
        file.write("123")
    shutil.move(file_path, file_path)
    os.remove(file_path)


if __name__ == '__main__':
    start()
    pass
    pass
