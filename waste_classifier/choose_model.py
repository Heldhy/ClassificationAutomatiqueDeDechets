from tensorflow.python.keras import Input
from tensorflow.python.keras.applications.mobilenet import MobileNet
from tensorflow.python.keras.applications.resnet import ResNet50
from tensorflow.python.keras.applications.vgg16 import VGG16
from tensorflow.python.keras.layers import Dense, Conv2D, MaxPooling2D, BatchNormalization, Dropout, Flatten
from tensorflow.python.keras.models import Sequential

from waste_classifier import NB_CLASSES, HEIGHT, WIDTH


def bootleneck_feature_extractor(train_generator):
    model = VGG16()
    return None

def return_mobilenet():
    model = Sequential()
    new_input = Input(shape=(HEIGHT, WIDTH, 3))
    model.add(MobileNet(include_top=False, weights='imagenet', input_tensor=new_input, pooling="avg"))
    model.add(Dense(NB_CLASSES, activation='softmax'))
    return model


def return_vgg16_freezed():
    model = Sequential()
    new_input = Input(shape=(HEIGHT, WIDTH, 3))
    model.add(VGG16(include_top=False, input_tensor=new_input, weights="imagenet", pooling='avg'))
    model.add(Dense(NB_CLASSES, activation='softmax'))
    model.layers[0].trainable = False
    return model


def return_vgg16():
    model = Sequential()
    new_input = Input(shape=(HEIGHT, WIDTH, 3))
    model.add(VGG16(include_top=False, input_tensor=new_input, weights="imagenet", pooling='avg'))
    model.add(Dense(NB_CLASSES, activation='softmax'))
    return model


def return_resnet50():
    model = Sequential()
    model.add(ResNet50(include_top=False, pooling='avg', weights='imagenet'))
    model.add(Dense(512, activation='relu'))
    model.add(Dense(NB_CLASSES, activation='softmax'))
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
    model.add(Dense(NB_CLASSES, activation='softmax'))
    return model


def get_model(model):
    if (model == "resnet50"):
        return return_resnet50()
    if (model == "crafted_model"):
        return return_crafted_model()
    if (model == "vgg16"):
        return return_vgg16()
    if (model == "vgg16_freezed"):
        return return_vgg16_freezed()
    if (model == "mobilenet"):
        return return_mobilenet()
    return None
