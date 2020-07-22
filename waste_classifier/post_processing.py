import numpy as np
from scipy.optimize import minimize_scalar
from sklearn.metrics import accuracy_score, recall_score
from tensorflow.keras.utils import to_categorical

from waste_classifier import CLASSES, CLASSES_TO_TRASH, AN_PRO_PROCESS_TALE, WASTE_TYPE
from waste_classifier.soft_recall import soft_recall_function


def convert_to_trash(prediction: np.ndarray, y: np.ndarray) -> (np.ndarray, np.ndarray):
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


def predict_and_convert_to_trash(model, x: np.ndarray, y: np.ndarray) -> (np.ndarray, np.ndarray):
    prediction = model.predict(x)
    return convert_to_trash(prediction, y)


def evaluate_post_processed_prediction(model, x: np.ndarray, y: np.ndarray) -> (float, float):
    y_predicted, y_true = predict_and_convert_to_trash(model, x, y)
    accuracy = accuracy_score(y_true, y_predicted)
    recall = recall_score(y_true, y_predicted, average='macro')
    print("accuracy : " + str(accuracy))
    print("recall : " + str(recall))
    return accuracy, recall


def compute_trash_classification_with_threshold(prediction: np.ndarray, y: np.ndarray, threshold=0):
    y_predicted = []
    y_true = []
    for index, current_prediction in enumerate(prediction):
        prediction_list = current_prediction.tolist()
        top_prediction_index = np.argmax(current_prediction)
        second_prediction_index = prediction_list.index(
            max(prediction_list[:top_prediction_index] + prediction_list[top_prediction_index + 1:]))
        top_prediction = CLASSES_TO_TRASH[CLASSES[top_prediction_index]]
        second_prediction = CLASSES_TO_TRASH[CLASSES[second_prediction_index]]
        expected_prediction = CLASSES_TO_TRASH[CLASSES[np.argmax(y[index])]]
        if top_prediction != second_prediction and current_prediction[top_prediction_index] < threshold:
            top_prediction = CLASSES_TO_TRASH["trash"]
        y_predicted.append(top_prediction)
        y_true.append(expected_prediction)
    y_predicted = to_categorical(np.array(y_predicted), 3)
    y_true = to_categorical(np.array(y_true), 3)
    soft_recall = soft_recall_function(y_predicted, y_true, AN_PRO_PROCESS_TALE, WASTE_TYPE).soft_recall
    accuracy = accuracy_score(y_true, y_predicted)
    recall = recall_score(y_true, y_predicted, average='macro')
    print("accuracy : " + str(accuracy))
    print("recall : " + str(recall))
    print("soft recall : " + str(soft_recall))
    return y_predicted, y_true


def compute_best_threshold_for_dispatching_of_trash(prediction: np.ndarray, y: np.ndarray):
    def compute_score(threshold):
        y_predicted = []
        y_true = []
        for index, current_prediction in enumerate(prediction):
            prediction_list = current_prediction.tolist()
            top_prediction_index = np.argmax(current_prediction)
            second_prediction_index = prediction_list.index(
                max(prediction_list[:top_prediction_index] + prediction_list[top_prediction_index + 1:]))
            top_prediction = CLASSES_TO_TRASH[CLASSES[top_prediction_index]]
            second_prediction = CLASSES_TO_TRASH[CLASSES[second_prediction_index]]
            expected_prediction = CLASSES_TO_TRASH[CLASSES[np.argmax(y[index])]]
            if top_prediction != second_prediction and current_prediction[top_prediction_index] < threshold:
                top_prediction = CLASSES_TO_TRASH["trash"]
            y_predicted.append(top_prediction)
            y_true.append(expected_prediction)
        y_predicted = to_categorical(np.array(y_predicted), 3)
        y_true = to_categorical(np.array(y_true), 3)
        score = - soft_recall_function(y_predicted, y_true, AN_PRO_PROCESS_TALE, WASTE_TYPE).soft_recall
        return score

    return minimize_scalar(compute_score, bounds=(0, 1), method='bounded').x
