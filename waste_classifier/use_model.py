import numpy as np
from matplotlib.pyplot import imread
from tensorflow.python.keras.applications.mobilenet import preprocess_input
from tensorflow.python.ops.nn_ops import softmax

from calibration import get_logits_friendly_model
from pre_processing import make_image_square
from waste_classifier import HEIGHT, WIDTH, CLASSES, WASTE_TYPE, CLASSES_TO_TRASH


def return_trash_label(label_number: int) -> int:
    """
    :param label_number: the label of each class
        - cardboard: 0
        - glass: 1
        - metal: 2
        - paper: 3
        - plastic: 4
        - trash: 5
    :return: the new label corresponding to recyclable, verre or non recyclable
    """
    return CLASSES_TO_TRASH[CLASSES[label_number]]


def predict_image(model, path: str, temperature_scaling: float = 1.0) -> str:
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
