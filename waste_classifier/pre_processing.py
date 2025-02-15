from pathlib import Path

from cv2.cv2 import INTER_AREA, resize
from matplotlib.pyplot import imread
from numpy import uint8, ndarray, array, random, full
from tensorflow.keras.utils import to_categorical
from tensorflow.python.keras.applications.mobilenet import preprocess_input

from waste_classifier import CLASS_TO_INDEX, BASE_DIR


def no_pre_processing(data: ndarray) -> ndarray:
    return data


def make_image_square(img: ndarray, min_size: int = 224) -> ndarray:
    height = img.shape[0]
    width = img.shape[1]
    largest_dimension = max(img.shape)
    square_shaped_image = full((largest_dimension, largest_dimension, 3), 255, uint8)
    top_left_corner_x, top_left_corner_y = (largest_dimension - width) // 2, (largest_dimension - height) // 2
    square_shaped_image[top_left_corner_y:height + top_left_corner_y, top_left_corner_x:width + top_left_corner_x] = img
    return resize(square_shaped_image, (min_size, min_size), interpolation=INTER_AREA)


def get_preprocessed_data(path: Path, pre_processing_function) -> (ndarray, ndarray):
    x_data = []
    y_data = []
    path_to_data = Path(path)
    files = sorted(path_to_data.iterdir())
    for directories in files:
        for images in sorted(directories.iterdir()):
            img = imread(images)
            img = make_image_square(img)
            x_data.append(pre_processing_function(img))
            y_data.append(CLASS_TO_INDEX[directories.name])
    return array(x_data), array(y_data)


def get_data(directory:  Path) -> (ndarray, ndarray, ndarray, ndarray):
    train_folder = directory / 'train'
    test_folder = directory / 'test'
    x_train, y_train = get_preprocessed_data(train_folder, no_pre_processing)
    x_test, y_test = get_preprocessed_data(test_folder, preprocess_input)
    return x_train, y_train, x_test, y_test


def pre_processing(directory: Path = None) -> (ndarray, ndarray, ndarray, ndarray):
    if directory is None:
        directory = BASE_DIR
    x_train, y_train, x_test, y_test = get_data(directory)
    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test)
    random_permutation_to_apply = random.permutation(len(y_train))
    x_train, y_train = x_train[random_permutation_to_apply], y_train[random_permutation_to_apply]
    return x_train, y_train, x_test, y_test
