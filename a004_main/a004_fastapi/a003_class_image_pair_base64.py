from pydantic import BaseModel


class ImagePairBase64Request(BaseModel):
    image_0: str
    image_1: str
