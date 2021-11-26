import tensorflow as tf

from pickle import load
from datetime import datetime

from utils.config import Config
from utils.logger import get_logger
LOG = get_logger('PredictItos')


class PredictItos:
    def __init__(self, config):
        self.units = Config.from_json(config).model.units
        LOG.info(f'Init: {datetime.now().time()}')
        self.feature_model =tf.keras.models.load_model(
            'saved_model/inceptionv3', compile=False)
        LOG.info(f'inception loaded: {datetime.now().time()}')
        self.encoder = tf.keras.models.load_model(
            'saved_model/encoder', compile=False)
        LOG.info(f'Encoder loaded: {datetime.now().time()}')
        self.decoder = tf.keras.models.load_model(
            'saved_model/decoder', compile=False)
        LOG.info(f'Decoder loaded {datetime.now().time()}')
        self.tokenizer = None
        with open("saved_model/tokenizer.pkl", "rb") as f:
            self.tokenizer = load(f)
        LOG.info(f'Tokenizer loaded: {datetime.now().time()}')

    def preprocess(self, image):
        image = tf.image.resize(image, (299, 299))
        image = tf.keras.applications.inception_v3.preprocess_input(image)
        preprocessed_img = tf.expand_dims(image, 0)
        return preprocessed_img

    def get_image_tensor(self, image):
        input_img = self.preprocess(image)
        img_tensor_val = self.feature_model(input_img)
        img_tensor_val = tf.reshape(
            img_tensor_val, (img_tensor_val.shape[0], -1, img_tensor_val.shape[3]))
        return img_tensor_val
    
    def filter_text(self, text):
        filt=['<start>','<unk>','<end>']
        text = [word for word in text if word not in filt]
        text = ' '.join(text)
        return text

    def predict(self, image=None):
        img_tensor_val = self.get_image_tensor(image)
        features = self.encoder(img_tensor_val)
        dec_input = tf.expand_dims([self.tokenizer.word_index['<start>']], 0)
        hidden = tf.zeros((1, self.units))
        result = []

        for i in range(39):  # todo:correct max length - add programmaticaly
            predictions, state, attention_weights = self.decoder(dec_input,
                                                                 features,
                                                                 hidden)
            predicted_id = tf.random.categorical(predictions, 1)[0][0].numpy()
            result.append(self.tokenizer.index_word[predicted_id])

            if self.tokenizer.index_word[predicted_id] == '<end>':
                return self.filter_text(result) 

            dec_input = tf.expand_dims([predicted_id], 0)

        return self.filter_text(result)
    
    def predict_service(self, image=None):
        image = tf.convert_to_tensor(image)
        caption = self.predict(image)
        return caption

