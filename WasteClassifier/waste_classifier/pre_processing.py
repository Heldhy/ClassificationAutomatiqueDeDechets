import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.utils import to_categorical
from variables import *


def no_preprocessing(data):
    return data


def make_square(img, min_size=224):
    s = max(img.shape[:2])
    f = np.full((s, s, 3), img.mean(), np.uint8)
    ax, ay = (s - img.shape[1]) // 2, (s - img.shape[0]) // 2
    f[ay:img.shape[0] + ay, ax:ax + img.shape[1]] = img
    return cv2.resize(f, (min_size, min_size), interpolation=cv2.INTER_AREA)


def get_preprocessed_data(path, preprocessing_function):
    x_data = []
    y_data = []
    files_in_train = sorted(os.listdir(path))
    for i in files_in_train:
        for j in sorted(os.listdir(path + i)):
            img = plt.imread(path + i + '/' + j)
            img = make_square(img)
            x_data.append(preprocessing_function(img))
            y_data.append(CLASS_TO_INDEX[i])
    return np.array(x_data), np.array(y_data)


def get_data():
    train_folder = BASE_DIR + 'train/'
    test_folder = BASE_DIR + 'test/'
    x_train, y_train = get_preprocessed_data(train_folder, no_preprocessing)
    x_test, y_test = get_preprocessed_data(test_folder, preprocess_input)
    return x_train, y_train, x_test, y_test


def preprocessing():
    x_train, y_train, x_test, y_test = get_data()
    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test)
    idx = np.random.permutation(len(y_train))
    x_train, y_train = x_train[idx], y_train[idx]
    return x_train, y_train, x_test, y_test
