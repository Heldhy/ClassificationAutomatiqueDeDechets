from waste_classifier.variables import *
import numpy as np
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import accuracy_score, recall_score


def convert_to_trash(model, x, y):
    prediction = model.predict(x)
    y_predicted = []
    y_true = []
    for i in range(len(prediction)):
        p = prediction[i].tolist()
        idx = p.index(max(p))
        true = CLASSES[y[i].tolist().index(1.0)]
        res = CLASSES[idx]
        y_true.append(CLASSES_TO_TRASH[true])
        y_predicted.append((CLASSES_TO_TRASH[res]))

    y_predicted = to_categorical(np.array(y_predicted), 3)
    y_true = to_categorical(np.array(y_true), 3)
    return y_predicted, y_true


def evaluate_postprocessed_model(model, x, y):
    y_predicted, y_true = convert_to_trash(model, x, y)
    accuracy = accuracy_score(y_true, y_predicted)
    recall = recall_score(y_true, y_predicted, average='macro')
    print("accuracy : " + str(accuracy))
    print("recall : " + str(recall))
    return accuracy, recall
