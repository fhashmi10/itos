import tensorflow as tf
import os
import numpy as np

from utils.logger import get_logger
LOG = get_logger('transfer')


class TransferInceptionV3:
    def __init__(self):
        # include_top = False will not include FC layer as last layer
        self.image_model = tf.keras.applications.InceptionV3(
            include_top=False, weights='imagenet')
        self.input = self.image_model.input
        self.hidden = self.image_model.layers[-1].output
        self.image_features_extract_model = tf.keras.Model(
            self.input, self.hidden)

    def save_features(self, images_folder, image_dataset):
        # avoid creating npy files multiple times
        npy_exists = False
        for fname in os.listdir(images_folder):
            if fname.endswith('.npy'):
                npy_exists = True
                LOG.info('Features are already saved.')
                break
        # create features npy files if none exists
        if not npy_exists:
            for img, path in image_dataset:
                batch_features = self.image_features_extract_model(img)
                batch_features = tf.reshape(batch_features,
                                            (batch_features.shape[0], -1, batch_features.shape[3]))

                for bf, p in zip(batch_features, path):
                    # first param is file name, second param is args (data)
                    np.save(p.numpy().decode("utf-8"), bf.numpy())
            LOG.info('Saved features successfully.')
