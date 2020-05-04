import numpy as np
from matplotlib.pyplot import imread

from pre_processing import make_square
from pre_processing import preprocess_input
from waste_classifier import HEIGHT, WIDTH, CLASSES, WASTE_TYPE


def predict_image(model, path):
    img = imread(path)
    img = preprocess_input(make_square(img))
    pred = model.predict(img.reshape((1, HEIGHT, WIDTH, 3)))
    prediction_list = pred.tolist()[0]
    max_index = np.argmax(pred)
    second_max_index = prediction_list.index(max(prediction_list[:max_index] + prediction_list[max_index + 1:]))
    trash = 0 if (max_index in {0, 2, 3, 4}) else 1 if max_index == 1 else 2
    print(CLASSES[max_index] + ": " + str(max(prediction_list) * 100)[:5] + "%")
    print(CLASSES[second_max_index] + ": " + str(prediction_list[second_max_index] * 100)[:5] + "%")
    print("--")
    return WASTE_TYPE[trash]
