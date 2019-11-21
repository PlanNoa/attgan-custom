from keras.applications import ResNet50
from keras.preprocessing.image import img_to_array
from keras.applications import imagenet_utils
from PIL import Image
import numpy as np
import flask
import io

app = flask.Flask(__name__)
model = None

def prepare_image(image, target):
    # 만약 이미지가 RGB가 아니라면, RGB로 변환해줍니다.
    if image.mode != "RGB":
        image = image.convert("RGB")

    # 입력 이미지 사이즈를 재정의하고 사전 처리를 진행합니다.
    image = image.resize(target)
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    image = imagenet_utils.preprocess_input(image)

    # 처리된 이미지를 반환합니다.
    return image