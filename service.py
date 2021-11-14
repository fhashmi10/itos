from prediction.predict_itos import PredictItos

import traceback
import gc

from configs.config import CFG
from flask import Flask, jsonify, request

app = Flask(__name__)

predictor = PredictItos(CFG)


@app.route('/predict', methods=["POST"])
def predict():
    data = request.json
    image = data['image']
    caption = predictor.predict_service(image)
    gc.collect()
    return caption


@app.route('/', methods=["GET"])
def index():
    try:
        return '<p>Use API: http://fh-itos.herokuapp.com/predict</p><'
    except:
        return '<p>Exception Occurred!!</p>'


@app.errorhandler(Exception)
def handle_exception(e):
    return jsonify(stackTrace=traceback.format_exc())


if __name__ == '__main__':
    app.run()
