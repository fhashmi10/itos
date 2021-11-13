from prediction.predict_itos import PredictItos
import os
import traceback
import tensorflow as tf

from configs.config import CFG
from flask import Flask, jsonify, request

app = Flask(__name__)

APP_ROOT = os.getenv('APP_ROOT', '/predict')
HOST = "localhost"
PORT_NUMBER = int(os.getenv('PORT_NUMBER', 8080))


predictor = PredictItos(CFG)


@app.route(APP_ROOT, methods=["POST"])
def predict():
    data = request.json
    image = data['image']
    image = tf.convert_to_tensor(data['image'])
    caption = predictor.predict(image)
    return caption


@app.errorhandler(Exception)
def handle_exception(e):
    return jsonify(stackTrace=traceback.format_exc())


if __name__ == '__main__':
    app.run(host=HOST, port=PORT_NUMBER)
