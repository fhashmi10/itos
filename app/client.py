import requests
from PIL import Image
import numpy as np

ENDPOINT_URL = "http://127.0.0.1:5000/predict"


def predict():
    image = np.asarray(Image.open(
        '../data/Images/667626_18933d713e.jpg')).astype(np.float32)
    data = {'image': image.tolist()}
    response = requests.post(ENDPOINT_URL, json=data)
    print(response.text)


if __name__ == "__main__":
    predict()
