import tensorflow as tf
from .attention import Attention


class Decoder(tf.keras.Model):
    def __init__(self, units, vocab_size, embed_dim):
        super(Decoder, self).__init__()
        self.units = units
        self.attention = Attention(self.units)
        # define layers during initialization
        self.embed = tf.keras.layers.Embedding(vocab_size, embed_dim)
        self.gru = tf.keras.layers.GRU(
            self.units, return_sequences=True, return_state=True, recurrent_initializer='glorot_uniform')
        self.dense1 = tf.keras.layers.Dense(self.units)
        self.dense2 = tf.keras.layers.Dense(vocab_size)

    def call(self, x, features, hidden):
        # call attention model to get context vector & attention weights
        context_vector, attention_weights = self.attention(features, hidden)
        # embed input to shape (batch_size, 1, embedding_dim)
        embed = self.embed(x)
        # Concatenate your input with the context vector from attention layer. Shape (batch_size, 1, embedding_dim + embedding_dim)
        embed = tf.concat([tf.expand_dims(context_vector, 1), embed], axis=-1)
        # Extract the output & hidden state from GRU layer. Output shape (batch_size, max_length, hidden_size)
        output, state = self.gru(embed)
        output = self.dense1(output)
        # shape (batch_size * max_length, hidden_size)
        output = tf.reshape(output, (-1, output.shape[2]))
        # shape (batch_size * max_length, vocab_size)
        output = self.dense2(output)

        return output, state, attention_weights
