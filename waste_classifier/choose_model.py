from tensorflow.python.keras import Input
from tensorflow.python.keras.applications.mobilenet import MobileNet
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.models import Sequential

from waste_classifier import NB_CLASSES, HEIGHT, WIDTH


def add_classification_layer(base_model):
    model = Sequential()
    model.add(base_model)
    model.add(Dense(NB_CLASSES, activation='softmax'))
    return model


def return_frozen_mobilenet():
    shape = (HEIGHT, WIDTH, 3)
    new_input = Input(shape=shape)
    base_model = MobileNet(include_top=False, weights='imagenet', input_tensor=new_input, pooling="avg",
                           input_shape=shape)
    base_model.trainable = False
    return base_model
