# -*- coding: utf-8 -*-
"""Sreehari_task5.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1WunGWkv5BwebdycQeccs-kvZh2IHeqUv
"""

import tensorflow.keras.backend as K
import math
import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import EarlyStopping, LambdaCallback

class LRFinder:
    """
    Learning rate range test detailed in Cyclical Learning Rates for Training
    Neural Networks by Leslie N. Smith. The learning rate range test is a test
    that provides valuable information about the optimal learning rate. During
    a pre-training run, the learning rate is increased linearly or
    exponentially between two boundaries. The low initial learning rate allows
    the network to start converging and as the learning rate is increased it
    will eventually be too large and the network will diverge.
    """

    def __init__(self, model):
        self.model = model
        self.losses = []
        self.learning_rates = []
        self.best_loss = 1e9

    def on_batch_end(self, batch, logs):
        lr = K.get_value(self.model.optimizer.lr)
        self.learning_rates.append(lr)

        loss = logs['loss']
        self.losses.append(loss)

        if batch > 5 and (math.isnan(loss) or loss > self.best_loss * 4):
            self.model.stop_training = True
            return

        if loss < self.best_loss:
            self.best_loss = loss

        lr *= self.lr_mult
        K.set_value(self.model.optimizer.lr, lr)

    def find(self, dataset, start_lr, end_lr, epochs=1,
             steps_per_epoch=None, **kw_fit):
        if steps_per_epoch is None:
            raise Exception('To correctly train on the datagenerator,'
                            '`steps_per_epoch` cannot be None.'
                            'You can calculate it as '
                            '`np.ceil(len(TRAINING_LIST) / BATCH)`')

        self.lr_mult = (float(end_lr) /
                        float(start_lr)) ** (float(1) /
                                             float(epochs * steps_per_epoch))
        initial_weights = self.model.get_weights()

        original_lr = K.get_value(self.model.optimizer.lr)
        K.set_value(self.model.optimizer.lr, start_lr)

        callback = LambdaCallback(on_batch_end=lambda batch,
                                  logs: self.on_batch_end(batch, logs))

        self.model.fit(dataset,
                       epochs=epochs, callbacks=[callback], **kw_fit)
        self.model.set_weights(initial_weights)

        K.set_value(self.model.optimizer.lr, original_lr)

    def get_learning_rates(self):
        return(self.learning_rates)

    def get_losses(self):
        return(self.losses)

    def get_derivatives(self, sma):
        assert sma >= 1
        derivatives = [0] * sma
        for i in range(sma, len(self.learning_rates)):
            derivatives.append((self.losses[i] - self.losses[i - sma]) / sma)
        return derivatives

    def get_best_lr(self, sma, n_skip_beginning=10, n_skip_end=5):
        derivatives = self.get_derivatives(sma)
        best_der_idx = np.argmin(derivatives[n_skip_beginning:-n_skip_end])
        return self.learning_rates[n_skip_beginning:-n_skip_end][best_der_idx]

(ds_train, ds_test), ds_info = tfds.load(
    'mnist',
    split=['train', 'test'],
    shuffle_files=True,
    as_supervised=True,
    with_info=True,
)

def normalize_img(image, label):
  """Normalizes images: `uint8` -> `float32`."""
  return tf.cast(image, tf.float32) / 255., label

ds_train = ds_train.map(
    normalize_img, num_parallel_calls=tf.data.experimental.AUTOTUNE)
ds_train = ds_train.cache()
ds_train = ds_train.shuffle(ds_info.splits['train'].num_examples)
ds_train = ds_train.batch(128)
ds_train = ds_train.prefetch(tf.data.experimental.AUTOTUNE)

ds_test = ds_test.map(
    normalize_img, num_parallel_calls=tf.data.experimental.AUTOTUNE)
ds_test = ds_test.batch(128)
ds_test = ds_test.cache()
ds_test = ds_test.prefetch(tf.data.experimental.AUTOTUNE)

model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(input_shape=(28, 28)),
  tf.keras.layers.Dense(128,activation='relu'),
  tf.keras.layers.Dense(10)
])
model.compile(
    optimizer=tf.keras.optimizers.Adam(0.001),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=[tf.keras.metrics.SparseCategoricalAccuracy()],
)

BATCH = 128

earlystop_callback = EarlyStopping(monitor='val_loss', min_delta=0.0001, patience=3)

num_epochs = 10 
model.fit(ds_train, epochs=num_epochs, validation_data=ds_test, callbacks=[earlystop_callback])
model.evaluate(ds_test)

