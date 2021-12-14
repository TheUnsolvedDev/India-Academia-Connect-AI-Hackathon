import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import os
import json

physical_devices = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)

model = tf.keras.models.load_model('model.h5')
path = './test/'

test_non_text = [10, 12, 14, 16, 18, 2, 31, 32, 33, 34,
                 35, 36, 37, 38, 39, 4, 41, 43, 45, 47,
                 49, 56, 57, 58, 59, 6, 60, 64, 65, 66, 74,
                 75, 76, 77, 78, 79, 8, 80, 85, 86, 87, 88, 89,
                 95, 96, 97, 98]
test_text = [1, 11, 13, 15, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 3,
             30, 42, 46, 48, 5, 50, 51, 52, 53, 54, 55, 61, 62, 63, 67, 68, 69,
             7, 70, 71, 72, 73, 81, 82, 83, 84, 90, 9, 91, 92, 93, 94, 44, 17, 40]


def write_json(filename, data):
    with open(filename, 'w') as f:
        json.dump(data, f)


def read_json(filename):
    with open(filename, 'r') as f:
        data = json.load(f)
    return data


def read_img(img):
    img = tf.io.read_file(img)
    img = tf.image.decode_jpeg(img, channels=3)
    return img


def generate_sample_file(filename):
    res = {}
    for i in range(1, 99):
        image_no = str(i) + '.jpg'
        image = read_img('test/'+image_no)
        result = model.predict(tf.expand_dims(image, axis=0))
        if result > 0.5:
            res[image_no] = 1
        else:
            res[image_no] = 0
    write_json(filename, res)


if __name__ == '__main__':
    generate_sample_file('./sample_result1.json')
