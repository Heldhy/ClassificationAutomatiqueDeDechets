from typing import Dict, List

from waste_classifier.adapted_confusion_matrix import AdaptedConfusionMatrix


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

