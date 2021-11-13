import tensorflow as tf


class Attention(tf.keras.Model):
    def __init__(self, units):
        super(Attention, self).__init__()
        self.units = units
        # define layers during initialization
        self.W1 = tf.keras.layers.Dense(self.units)
        self.W2 = tf.keras.layers.Dense(self.units)
        self.V = tf.keras.layers.Dense(1)

    def call(self, features, hidden):
        # expand hidden to shape (batch_size, 1, hidden_size)
        hidden_with_time_axis = tf.expand_dims(hidden, 1)
        attention_hidden_layer = tf.nn.tanh(
            self.W1(features) + self.W2(hidden_with_time_axis))
        # build score funciton to shape (batch_size, 8*8, units)
        score = self.V(attention_hidden_layer)
        # extract attention weights with shape (batch_size, 8*8, 1)
        attention_weights = tf.nn.softmax(score, axis=1)
        # create the context vector with shape (batch_size, 8*8,embedding_dim)
        context_vector = attention_weights * features
        # reduce the shape to (batch_size, embedding_dim)
        context_vector = tf.reduce_sum(context_vector, axis=1)

        return context_vector, attention_weights
