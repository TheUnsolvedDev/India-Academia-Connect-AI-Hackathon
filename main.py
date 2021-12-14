import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.python.ops.gen_batch_ops import Batch

import dataset

physical_devices = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)

train_folder = './training/'

IMAGE_HEIGHT = 64
IMAGE_WIDTH = 64
N_CHANNELS = 3
N_CLASSES = 1

TRAIN_BATCH_SIZE = 128
TEST_BATCH_SIZE = 32


class CNN_model:
    def __init__(self, train_generator, test_generator) -> None:
        self.train_generator = train_generator
        self.test_generator = test_generator

        self.callbacks = [
            tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=7),
            tf.keras.callbacks.ModelCheckpoint(
                filepath='model.h5', monitor='val_loss', save_best_only=True),
            tf.keras.callbacks.TensorBoard(
                log_dir='./logs', histogram_freq=1, write_graph=True),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss', factor=0.1, patience=4, verbose=1, mode='auto', min_delta=0.0001, cooldown=0, min_lr=0)
        ]

    # lenet 5
    def model(self):
        inputs = tf.keras.layers.Input(
            shape=(IMAGE_HEIGHT, IMAGE_WIDTH, N_CHANNELS))
        conv1 = tf.keras.layers.Conv2D(filters=6, kernel_size=(
            5, 5), activation='relu', kernel_initializer='he_uniform')(inputs)
        pool1 = tf.keras.layers.MaxPool2D(pool_size=(2, 2))(conv1)
        conv2 = tf.keras.layers.Conv2D(filters=16, kernel_size=(
            5, 5), activation='relu', kernel_initializer='he_uniform')(pool1)
        pool2 = tf.keras.layers.MaxPool2D(pool_size=(2, 2))(conv2)
        flat = tf.keras.layers.Flatten()(pool2)
        dense1 = tf.keras.layers.Dense(
            units=120, activation='relu', kernel_initializer='he_uniform')(flat)
        dense2 = tf.keras.layers.Dense(
            units=84, activation='relu', kernel_initializer='he_uniform')(dense1)
        outputs = tf.keras.layers.Dense(
            units=N_CLASSES, activation='sigmoid')(dense2)
        model = tf.keras.Model(inputs=inputs, outputs=outputs)
        return model

    def train(self):
        cnn_mdl = self.model()
        cnn_mdl.compile(optimizer='adam',
                        loss='binary_crossentropy', metrics=['accuracy'])
        cnn_mdl.summary()

        history = cnn_mdl.fit(self.train_generator, callbacks=self.callbacks,
                              epochs=50, validation_data=self.test_generator)
        self.plot_metrics(history.history)
        return history.history

    def plot_metrics(self, history):
        plt.plot(history['accuracy'])
        plt.plot(history['val_accuracy'])
        plt.title('model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.show()

        plt.plot(history['loss'])
        plt.plot(history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.show()

        return history


if __name__ == '__main__':
    train_gen = dataset.DataGenerator(
        batch_size=TRAIN_BATCH_SIZE, shuffle=True)
    test_gen = dataset.DataGenerator(batch_size=TEST_BATCH_SIZE, shuffle=False)
    classifier = CNN_model(train_gen, test_gen)
    classifier.train()
