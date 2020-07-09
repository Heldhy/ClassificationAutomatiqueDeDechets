from typing import Dict, List

from numpy import argmax, ndarray

from confusion_soft_recall import ConfusionSoftRecall


def acceptable_dict_to_matrix(acceptable_negative_table: Dict[str, List[str]], index_to_class_table: List[str]):
    acceptable_matrix = {}
    for current_class in index_to_class_table:
        acceptable_matrix[current_class] = {}
        for loop_class in index_to_class_table:
            acceptable_matrix[current_class][loop_class] = loop_class in acceptable_negative_table[current_class]
    return acceptable_matrix


def get_class(prediction: ndarray, index_to_class_table: List[str]):
    return index_to_class_table[argmax(prediction)]


def soft_recall_function(ypred: ndarray, ytrue: ndarray, acceptable_negative_table: Dict[str, List[str]],
                         index_to_class_table: List[str]):
    is_acceptable_negative_matrix = acceptable_dict_to_matrix(acceptable_negative_table, index_to_class_table)
    confusion_soft_recall = ConfusionSoftRecall(index_to_class_table)

    nb_of_true_samples_for_classes = {}
    for current_class in index_to_class_table:
        nb_of_true_samples_for_classes[current_class] = 0

    for index, current_prediction in enumerate(ypred):
        predicted_label = get_class(current_prediction, index_to_class_table)
        true_label = index_to_class_table[argmax(ytrue[index])]

        nb_of_true_samples_for_classes[true_label] += 1

        if predicted_label == true_label:
            confusion_soft_recall.confusion_matrix_list[true_label].true_positive += 1
        else:
            confusion_soft_recall.confusion_matrix_list[predicted_label].false_positive += 1
            if is_acceptable_negative_matrix[true_label][predicted_label]:
                confusion_soft_recall.confusion_matrix_list[true_label].acceptable_negative += 1
            else:
                confusion_soft_recall.confusion_matrix_list[true_label].false_negative += 1

    confusion_soft_recall.compute(nb_of_true_samples_for_classes)
    return confusion_soft_recall


def soft_recall_from_model(model, x_test: ndarray, y_true: ndarray, acceptable_negative_table: Dict[str, List[str]],
                           index_to_class_table: List[str]):
    y_pred = model.predict(x_test)
    return soft_recall_function(y_pred, y_true, acceptable_negative_table, index_to_class_table)
