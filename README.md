# 环境配置
使用Python版本3.12。创建新conda环境后运行如下安装命令。

Linux:
```
pip3 install torch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2 --index-url https://download.pytorch.org/whl/cu121
pip3 install facenet-pytorch deepface onnxruntime-gpu opencv-python seaborn fastapi 'uvicorn[standard]' colored_traceback python-multipart tf-keras scikit-learn colorama natsort retinaface-pytorch
pip3 install insightface
```

WINDOWS:
```
pip3 install torch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2 --index-url https://download.pytorch.org/whl/cu121
pip3 install facenet-pytorch deepface onnxruntime-gpu opencv-python seaborn fastapi 'uvicorn[standard]' colored_traceback python-multipart tf-keras scikit-learn colorama natsort retinaface-pytorch
pip3 install insightface
```

# 启动说明
1. cd进入项目目录
2. 运行如下命令，启动服务
 ```
 export PYTHONPATH=/home/mkx/_Search/projects/a003_face:$PYTHONPATH
 python ./a002_main/a004_fastapi/a001_main.py
 ```
 若要挂在后台运行，可将第二条命令改为 "nohup python ./a002_main/a004_fastapi/a001_main.py > output.log 2>&1 &"