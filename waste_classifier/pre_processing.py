from pathlib import Path

import numpy as np
from cv2.cv2 import INTER_AREA, resize
from matplotlib.pyplot import imread
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.utils import to_categorical

from waste_classifier import CLASS_TO_INDEX, BASE_DIR


def no_pre_processing(data):
    return data


def make_square(img, min_size=224):
    height = img.shape[0]
    width = img.shape[1]
    largest_dimension = max(img.shape)
    squared_image = np.full((largest_dimension, largest_dimension, 3), img.mean(), np.uint8)
    top_left_corner_x, top_left_corner_y = (largest_dimension - width) // 2, (largest_dimension - height) // 2
    squared_image[top_left_corner_y:height + top_left_corner_y, top_left_corner_x:width + top_left_corner_x] = img
    return resize(squared_image, (min_size, min_size), interpolation=INTER_AREA)


def get_preprocessed_data(path, pre_processing_function):
    x_data = []
    y_data = []
    p = Path(path)
    files = sorted(p.iterdir())
    for i in files:
        for j in sorted(i.iterdir()):
            img = imread(j)
            img = make_square(img)
            x_data.append(pre_processing_function(img))
            y_data.append(CLASS_TO_INDEX[i.name])
    return np.array(x_data), np.array(y_data)


def get_data():
    train_folder = BASE_DIR / 'train'
    test_folder = BASE_DIR / 'test'
    x_train, y_train = get_preprocessed_data(train_folder, no_pre_processing)
    x_test, y_test = get_preprocessed_data(test_folder, preprocess_input)
    return x_train, y_train, x_test, y_test


def pre_processing():
    x_train, y_train, x_test, y_test = get_data()
    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test)
    idx = np.random.permutation(len(y_train))
    x_train, y_train = x_train[idx], y_train[idx]
    return x_train, y_train, x_test, y_test
