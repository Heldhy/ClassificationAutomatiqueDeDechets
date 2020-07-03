import numpy as np
from matplotlib.pyplot import imread
from tensorflow.python.keras.applications.mobilenet import preprocess_input
from tensorflow.python.ops.nn_ops import softmax

from calibration import get_logits_friendly_model
from pre_processing import make_image_square
from waste_classifier import HEIGHT, WIDTH, CLASSES, WASTE_TYPE


def return_trash_label(previous_label):
    if previous_label in {0, 2, 3, 4}:
        return 0
    if previous_label == 1:
        return 1
    return 2


def predict_image(model, path, temperature_scaling=1):
    source_img = imread(path)
    img = preprocess_input(make_image_square(source_img))
    logit_model = get_logits_friendly_model(model)
    logits = logit_model.predict(img.reshape((1, HEIGHT, WIDTH, 3)))
    calibrated_logits = logits / temperature_scaling
    prediction = softmax(calibrated_logits).numpy()
    prediction_list = prediction.tolist()[0]
    max_index = np.argmax(prediction)
    second_max_index = prediction_list.index(max(prediction_list[:max_index] + prediction_list[max_index + 1:]))
    trash = return_trash_label(max_index)
    print(CLASSES[max_index] + ": " + str(round(max(prediction_list) * 100, 2)) + "%")
    print(CLASSES[second_max_index] + ": " + str(round(prediction_list[second_max_index] * 100, 2)) + "%")
    return WASTE_TYPE[trash]
