import tensorflow as tf


class TokenizerModel():
    def __init__(self, token_config):
        self.tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=token_config.vocab_size,
                                                               oov_token="<unk>",
                                                               filters='!"#$%&()*+.,-/:;=?@[\]^_`{|}~')

    def get_captions_vector(self, all_captions):
        all_captions = ['<start> '+cap+' <end>' for cap in all_captions]
        self.tokenizer.fit_on_texts(all_captions)
        # Create word-to-index and index-to-word mappings.
        self.tokenizer.word_index['<pad>'] = 0
        self.tokenizer.index_word[0] = '<pad>'
        # Convert text to sequence
        annotations_seq = self.tokenizer.texts_to_sequences(all_captions)
        cap_vector = tf.keras.preprocessing.sequence.pad_sequences(
            annotations_seq, padding='post')
            
        return cap_vector, self.tokenizer
