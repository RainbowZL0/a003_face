from flask import Flask, request, jsonify
from pydantic import BaseModel, ValidationError

from a003_fastapi.a002_my_fastapi_processor import MyFastapiProcessor
from a003_fastapi.a003_class_image_pair_base64 import ImagePairBase64Request

app = Flask(__name__)
my_fastapi_processor = MyFastapiProcessor()


@app.route("/facecomparsion", methods=["POST"])
def face_comparison():
    try:
        # 从请求中获取JSON数据并验证
        data = request.get_json()
        validated_data = ImagePairBase64Request(**data)

        return my_fastapi_processor.get_image_pair_and_verify_base64_version(
            image_0=validated_data.image_0,
            image_1=validated_data.image_1
        )
    except ValidationError as e:
        return jsonify({"error": e.errors()}), 422


def run_flask_server():
    app.run(
        host="0.0.0.0",
        port=8092,
        threaded=False  # 确保单线程处理请求
    )


if __name__ == "__main__":
    run_flask_server()