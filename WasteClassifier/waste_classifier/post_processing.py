from variables import *
import numpy as np
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import accuracy_score, recall_score


def convert_to_trash(prediction,  y):
    y_predicted = []
    y_true = []
    for i in range(len(prediction)):
        true = CLASSES[np.argmax(y[i])]
        res = CLASSES[np.argmax(prediction[i])]
        y_true.append(CLASSES_TO_TRASH[true])
        y_predicted.append((CLASSES_TO_TRASH[res]))
    y_predicted = to_categorical(np.array(y_predicted), 3)
    y_true = to_categorical(np.array(y_true), 3)
    return y_predicted, y_true


def predict_and_convert_to_trash(model, x, y):
    prediction = model.predict(x)
    return convert_to_trash(prediction, y)


def evaluate_post_processed_prediction(model, x, y):
    y_predicted, y_true = predict_and_convert_to_trash(model, x, y)
    accuracy = accuracy_score(y_true, y_predicted)
    recall = recall_score(y_true, y_predicted, average='macro')
    print("accuracy : " + str(accuracy))
    print("recall : " + str(recall))
    return accuracy, recall
