import tensorflow as tf

from .base_model import BaseModel
from .tokenizer_model import TokenizerModel
from transfer.transfer_inceptionV3 import TransferInceptionV3
from .encoder import Encoder
from .decoder import Decoder

from dataset.dataloader import DataLoader
from training.train_itos import TrainItos

from pickle import dump

from utils.logger import get_logger
LOG = get_logger('itosmodel')


class ItosModel(BaseModel):
    def __init__(self, config):
        super().__init__(config)
        self.tokenizer_model = TokenizerModel(self.config.tokenize)
        self.tokenizer = None
        self.transfer_model = TransferInceptionV3()
        self.encoder_model = Encoder(self.config.model.embedding_dim)
        self.decoder_model = Decoder(
            self.config.model.units, self.config.tokenize.vocab_size, self.config.model.embedding_dim)

        self.img_dataset = None
        self.all_img_path = None
        self.all_captions = None
        self.cap_vector = None

        self.batch_size = self.config.train.batch_size
        self.buffer_size = self.config.train.buffer_size
        self.epoches = self.config.train.epoches

        self.train_dataset = []
        self.test_dataset = []

    def process_data(self):
        # Loads and Preprocess data
        LOG.info(f'Loading {self.config.data.image_path} dataset...')
        self.all_img_path, self.all_captions, self.img_dataset = DataLoader.load_data(
            self.config.data)
        # Use transfer model to save features
        self.transfer_model.save_features(
            self.config.data.image_path, self.img_dataset)
        # Generate captions vector
        self.cap_vector, self.tokenizer = self.tokenizer_model.get_captions_vector(
            self.all_captions)
        # Save tokenizer
        with open("./saved_run/tokenizer.pkl", "wb") as f:
            dump(self.tokenizer, f)
        # Split in train and test
        self.train_dataset, self.test_dataset = DataLoader.split_data(self.all_img_path, self.cap_vector, self.batch_size,
                                                                      self.buffer_size)
        LOG.info(f'Train and test split is done successfully.')

    def train(self):
        """Compiles and trains the model"""
        LOG.info('Training started')
        optimizer = tf.keras.optimizers.Adam()
        loss_func = tf.keras.losses.SparseCategoricalCrossentropy(
            from_logits=True)
        metrics = tf.keras.metrics.SparseCategoricalAccuracy()

        trainer = TrainItos(self.tokenizer, self.encoder_model, self.decoder_model,
                            self.train_dataset, loss_func, optimizer, metrics, self.epoches)
        trainer.train()
