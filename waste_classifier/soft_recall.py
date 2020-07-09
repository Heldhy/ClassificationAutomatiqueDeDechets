from typing import Dict, List

from numpy import argmax, ndarray


def acceptable_dict_to_matrix(acceptable_negative_table, index_to_class_table):
    acceptable_matrix = {}
    for current_class in index_to_class_table:
        acceptable_matrix[current_class] = {}
        for loop_class in index_to_class_table:
            acceptable_matrix[current_class][loop_class] = loop_class in acceptable_negative_table[current_class]
    return acceptable_matrix


def get_class(prediction, index_to_class_table):
    return index_to_class_table[argmax(prediction)]


class AdaptedConfusionMatrix:
    def __init__(self, name: str, mode="adaptive"):
        """

        :param name: name of the class the confusion matrix is about
        :param mode: [adaptive, fixed] two possible modes for the alpha parameter:
                    - adaptive: the alpha parameter is computed in regard of the number of true positive, acceptable
                    negative and false positive for the current class
                    - fixed: the alpha parameter is a fixed value, at 0.8
        """
        self.name = name
        self.mode = mode
        self.true_positive = 0
        self.false_positive = 0
        self.false_negative = 0
        self.acceptable_negative = 0
        self.soft_recall = 0
        self.precision = 0
        self.recall = 0
        self.f1_score = 0

    def compute(self):
        if self.mode == "adaptive":
            if self.true_positive + self.acceptable_negative:
                alpha = max(1 / (self.true_positive + self.acceptable_negative),
                            self.true_positive / (self.true_positive + self.acceptable_negative))
            else:
                alpha = 0
        elif self.mode == "fixed":
            alpha = 0.8
        else:
            raise Exception("mode must be either 'adaptive' or 'fixed', not '{:d}'", self.mode)

        if (self.true_positive + self.acceptable_negative + self.false_negative) == 0:
            fraction = 1
        else:
            fraction = self.true_positive + self.acceptable_negative + self.false_negative
        self.soft_recall = (self.true_positive + alpha * self.acceptable_negative) / fraction
        self.recall = self.true_positive / fraction
        self.precision = 0 if (self.true_positive + self.false_positive) == 0 else self.true_positive / (
                self.true_positive + self.false_positive)
        self.f1_score = 0 if (self.precision + self.recall) == 0 else 2 * (self.recall * self.precision) / (
                self.precision + self.recall)

    def __str__(self):
        self.compute()
        return "{: <13}{: >6d}{: >6d}{: >6d}{: >10.2f}{: >10.2f}{: >10.2f}{: >10.2f}".format(self.name,
                                                                                             self.true_positive,
                                                                                             self.acceptable_negative,
                                                                                             self.false_negative,
                                                                                             self.precision,
                                                                                             self.recall, self.f1_score,
                                                                                             self.soft_recall)


class ConfusionSoftRecall:
    def __init__(self, index_to_class_table: List[str]):
        """

        :param index_to_class_table: a table which make the link between a class, its name and the prediction of a
        deep learning model
        """

        self.index_to_class_table = index_to_class_table
        self.weight_recall = 0
        self.soft_recall = 0
        self.recall = 0
        self.weight_soft_recall = 0
        self.confusion_matrix_list = {}
        for current_class in index_to_class_table:
            self.confusion_matrix_list[current_class] = AdaptedConfusionMatrix(current_class)

    def compute(self, nb_of_true_samples_for_classes: Dict[str, int]):
        """

        :param nb_of_true_samples_for_classes: a dictionary which map the number of true sample per class to
        the class name
        """
        unweighted_sum_of_recall = 0
        sum_of_recall = 0
        sum_of_soft_recall = 0
        unweighted_sum_of_soft_recall = 0
        sum_of_elements = 0
        for current_class in self.index_to_class_table:
            self.confusion_matrix_list[current_class].compute()
            unweighted_sum_of_recall += nb_of_true_samples_for_classes[current_class] * self.confusion_matrix_list[
                current_class].recall
            sum_of_recall += self.confusion_matrix_list[current_class].recall
            sum_of_soft_recall += self.confusion_matrix_list[current_class].soft_recall
            unweighted_sum_of_soft_recall += nb_of_true_samples_for_classes[current_class] * self.confusion_matrix_list[
                current_class].soft_recall
            sum_of_elements += nb_of_true_samples_for_classes[current_class]

        nb_classes = len(self.index_to_class_table)
        self.soft_recall = sum_of_soft_recall / nb_classes
        self.weight_soft_recall = unweighted_sum_of_soft_recall / sum_of_elements
        self.recall = sum_of_recall / nb_classes
        self.weight_recall = unweighted_sum_of_recall / sum_of_elements

    def __str__(self):
        returned_string = 17 * " " + "TP    AN    FN   Precision   Recall  F1 score   SoftRecall\n"
        for current_class in self.index_to_class_table:
            returned_string += str(self.confusion_matrix_list[current_class]) + "\n"
        returned_string += "--\nSoftRecall\t\t\t" + str(round(self.soft_recall, 2))
        returned_string += "\nWeightedSoftRecall\t" + str(round(self.weight_soft_recall, 2))
        return returned_string


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
