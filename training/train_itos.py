import os
import tensorflow as tf

from utils.logger import get_logger

LOG = get_logger('trainer')


class TrainItos:
    def __init__(self, tokenizer, units, encoder, decoder, input, loss_fn, optimizer, metric, epochs):
        self.tokenizer = tokenizer
        self.units = units
        self.encoder = encoder
        self.decoder = decoder
        self.input = input
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.metric = metric
        self.epochs = epochs

        self.checkpoint = tf.train.Checkpoint(encoder=encoder,
                                              decoder=decoder,
                                              optimizer=optimizer)
        self.checkpoint_manager = tf.train.CheckpointManager(
            self.checkpoint, './checkpoints', max_to_keep=3)

    def loss_function(self, real, pred):
        mask = tf.math.logical_not(tf.math.equal(real, 0))
        loss_ = self.loss_fn(real, pred)
        mask = tf.cast(mask, dtype=loss_.dtype)
        loss_ *= mask
        return tf.reduce_mean(loss_)

    def train_step(self, img_tensor, target):
        loss = 0
        hidden = tf.zeros((1, self.units))
        dec_input = tf.expand_dims(
            [self.tokenizer.word_index['<start>']] * target.shape[0], 1)

        with tf.GradientTape() as tape:
            features = self.encoder(img_tensor)

            for i in range(1, target.shape[1]):
                # passing the features through the decoder
                predictions, hidden, _ = self.decoder(
                    dec_input, features, hidden)
                loss += self.loss_function(target[:, i], predictions)
                # using teacher forcing
                dec_input = tf.expand_dims(target[:, i], 1)

            trainable_variables = self.encoder.trainable_variables + \
                self.decoder.trainable_variables
            gradients = tape.gradient(loss, trainable_variables)
            self.optimizer.apply_gradients(zip(gradients, trainable_variables))

        return loss

    def train(self):
        self.checkpoint.restore(self.checkpoint_manager.latest_checkpoint)
        if self.checkpoint_manager.latest_checkpoint:
            LOG.info("Restored from {}".format(self.checkpoint_manager.latest_checkpoint))
            for batch, (img_tensor, target) in enumerate(self.input):
                batch_loss = self.train_step(img_tensor, target)
                break
        else:
            LOG.info("Initializing from scratch.")
            for epoch in range(self.epochs):
                LOG.info(f'Start epoch {epoch}')

                batch_loss = 0
                for batch, (img_tensor, target) in enumerate(self.input):
                    batch_loss = self.train_step(img_tensor, target)
                    LOG.info("Epoch %d - Loss at step %d: %.2f" %
                            (epoch+1, batch, batch_loss))
                    if batch % 100 == 0:
                        average_batch_loss = batch_loss.numpy() / \
                            int(target.shape[1])
                        LOG.info(
                            f'Epoch {epoch+1} - Batch {batch} - Loss {average_batch_loss:.4f}')

                if epoch % 5 == 0:
                    save_path = self.checkpoint_manager.save()
                    LOG.info("Saved checkpoint: {}".format(save_path))

                train_acc = self.metric.result()
                LOG.info("Training acc over epoch: %.4f" % (float(train_acc)))

                self.metric.reset_states()
        
        self.encoder.save('./saved_model/encoder/')
        LOG.info("Saved Encoder")
        
        self.decoder.save('./saved_model/decoder/')
        LOG.info("Saved Decoder")
