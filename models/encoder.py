import tensorflow as tf


class Encoder(tf.keras.Model):
    def __init__(self, embed_dim):
        super(Encoder, self).__init__()
        # define layers during initialization
        self.dense = tf.keras.layers.Dense(embed_dim)

    def call(self, features):
        # pass features through layers
        features = self.dense(features)
        features = tf.nn.relu(features)
        return features
