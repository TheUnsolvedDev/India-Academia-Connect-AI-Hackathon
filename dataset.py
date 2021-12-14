import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import os

train_folder = './training/'

IMAGE_HEIGHT = 64
IMAGE_WIDTH = 64
N_CHANNELS = 3
N_CLASSES = 1


class Dataset:
    def __init__(self, path=train_folder) -> None:
        items = os.listdir(path)

        self.train_data = []
        self.train_label = []

        for item in items:
            for images in os.listdir(path+item):
                image = tf.io.read_file(path+item+'/'+images)
                self.train_data.append(tf.image.decode_jpeg(image, channels=3))

                if item == 'background':
                    self.train_label.append(0)
                else:
                    self.train_label.append(1)

        self.train_data = np.array(self.train_data)
        self.train_label = np.array(self.train_label).reshape(-1, 1)

    def load_data(self):
        return self.train_data, self.train_label


class DataGenerator(tf.keras.utils.Sequence):
    def __init__(self, batch_size=32, n_channels=3, image_height=64, image_width=64, shuffle=True, n_class=N_CLASSES) -> None:
        self.batch_size = batch_size
        self.n_channels = n_channels
        self.image_height = image_height
        self.image_width = image_width
        self.n_class = n_class

        obj = Dataset()

        self.data, self.label = obj.load_data()

        self.indices = np.arange(len(self.data))
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        return int(np.floor(len(self.data) / self.batch_size))

    def __getitem__(self, index):
        indices = self.indices[index *
                               self.batch_size:(index + 1) * self.batch_size]
        return self.__data_generation(indices)

    def __data_generation(self, indices):
        x = np.empty((self.batch_size, self.image_height,
                     self.image_width, self.n_channels))
        y = np.empty((self.batch_size, self.n_class))

        for i, index in enumerate(indices):
            x[i] = self.data[index]
            y[i] = self.label[index]

        return x, y

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indices)


if __name__ == '__main__':
    obj = DataGenerator()
    obj.__getitem__(0)
    print(obj.__len__())