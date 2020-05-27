from sklearn.svm import SVC
from tensorflow.python.keras import Input
from tensorflow.python.keras.applications.mobilenet import MobileNet
from tensorflow.python.keras.applications.mobilenet_v2 import MobileNetV2
from tensorflow.python.keras.applications.resnet import ResNet50
from tensorflow.python.keras.applications.vgg16 import VGG16
from tensorflow.python.keras.layers import Dense, Conv2D, MaxPooling2D, BatchNormalization, Dropout, Flatten
from tensorflow.python.keras.models import Sequential

from waste_classifier import NB_CLASSES, HEIGHT, WIDTH


def return_svc():
    return SVC()


def build_hybrid_model(base_model):
    model = Sequential()
    model.add(base_model)
    #model.add(Dense(256, activation='relu'))
    model.add(Dense(NB_CLASSES, activation='softmax'))
    return model


def bootleneck_feature_extractor():
    new_input = Input(shape=(224, 224, 3))
    vgg = VGG16(include_top=False, input_tensor=new_input)
    return vgg

def return_frozen_mobilenet():
    shape = (HEIGHT, WIDTH, 3)
    new_input = Input(shape=shape)
    base_model = MobileNet(include_top=False, weights='imagenet', input_tensor=new_input, pooling="avg", input_shape=shape)
    base_model.trainable = False
    print(base_model.summary())
    return base_model

def return_mobilenet():
    model = Sequential()
    shape = (HEIGHT, WIDTH, 3)
    new_input = Input(shape=shape)
    model.add(MobileNet(include_top=False, weights='imagenet', input_tensor=new_input, pooling="avg", input_shape=shape))
    model.add(Dense(NB_CLASSES, activation='softmax'))
    return model

def return_mobilenetV2():
    model = Sequential()
    shape = (HEIGHT, WIDTH, 3)
    new_input = Input(shape=shape)
    model.add(MobileNetV2(include_top=False, weights='imagenet', input_tensor=new_input, pooling="avg", input_shape=shape))
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
    if (model == "svc"):
        return return_svc()
    if (model == "mobilenetV2"):
        return return_mobilenetV2()
    return None
