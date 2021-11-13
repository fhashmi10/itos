from inspect import trace
import tensorflow as tf
import traceback

from configs.config import CFG

from models.itos_model import ItosModel
from prediction.predict_itos import PredictItos


from utils.logger import get_logger
LOG = get_logger('main')


sample_filepath = './data/Images/667626_18933d713e.jpg'


def train():
    # build model
    model = ItosModel(CFG)
    # load data
    model.process_data()
    # train
    model.train()


def predict():
    predictor = PredictItos(CFG)
    image = tf.io.read_file(sample_filepath)
    image = tf.image.decode_image(image, channels=3)
    caption = predictor.predict(image)
    return caption


def run():
    # change train_mode to True if training needs to be performed
    train_mode = False
    try:
        if train_mode == True:
            train()

        # predict
        result = predict()
        print(result)

        # todo: metrics
    except Exception as e:
        LOG.error('An exception occured : ' + traceback.format_exc())


if __name__ == '__main__':
    run()
