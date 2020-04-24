from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, BatchNormalization, MaxPooling2D
from tensorflow.keras.applications.resnet50 import ResNet50
from variables import *


def return_resnet50():
    model = Sequential()
    model.add(ResNet50(include_top=False, pooling='avg', weights='imagenet'))
    model.add(Dense(512, activation='relu'))
    model.add(Dense(nb_classes, activation='softmax'))
    model.layers[0].trainable = False
    return model


def return_crafted_model():
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3), input_shape=(HEIGHT, WIDTH, 3), padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(BatchNormalization())
    model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(BatchNormalization())
    model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(BatchNormalization())
    model.add(Conv2D(32, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(BatchNormalization())
    model.add(Dropout(0.2))
    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(nb_classes, activation='softmax'))
    return model


def get_model(model):
    if (model == "resnet50"):
        return return_resnet50()
    if (model == "crafted_model"):
        return return_crafted_model()
    return None
