from numpy import argmax


def acceptable_dict_to_matrix(d, classes):
    acceptable_matrix = {}
    for current_class in classes:
        acceptable_matrix[current_class] = {}
        for loop_class in classes:
            acceptable_matrix[current_class][loop_class] = loop_class in d[current_class]
    return acceptable_matrix


def get_class(y, classes):
    idx = argmax(y)
    return classes[idx]


class AdaptedConfusionMatrix:
    def __init__(self, name, mode="fluide"):
        self.name = name
        self.TP = 0
        self.FP = 0
        self.FN = 0
        self.AN = 0
        self.SoftRecall = 0
        self.Precision = 0
        self.Recall = 0
        self.F1 = 0
        self.mode = mode

    def soft_recall(self):
        if (self.mode == "fluide"):
            if ((self.TP + self.AN)):
                alpha = max(1 / (self.TP + self.AN), self.TP / (self.TP + self.AN))
            else:
                alpha = 0
        else:
            alpha = 0.8

        fraction = 1 if (self.TP + self.AN + self.FN) == 0 else self.TP + self.AN + self.FN
        self.SoftRecall = (self.TP + alpha * self.AN) / fraction
        self.Recall = self.TP / fraction
        self.Precision = 0 if (self.TP + self.FP) == 0 else self.TP / (self.TP + self.FP)
        self.F1 = 0 if (self.Precision + self.Recall) == 0 else 2 * (self.Recall * self.Precision) / (
                self.Precision + self.Recall)

    def __str__(self):
        if (self.SoftRecall == 0):
            self.soft_recall()
        r = "{: <13}{: >6d}{: >6d}{: >6d}{: >10.2f}{: >10.2f}{: >10.2f}{: >10.2f}".format(self.name, self.TP, self.AN,
                                                                                          self.FN, self.Precision,
                                                                                          self.Recall, self.F1,
                                                                                          self.SoftRecall)
        return r


class ConfusionSoftRecall:
    def __init__(self, classes):
        self.nb_of_true_samples_for_classes = 0
        self.class_name = classes
        self.confusion_matrix_list = {}
        self.WeightRecall = 0
        self.SoftRecall = 0
        self.Recall = 0
        self.WeightSoftRecall = 0
        for current_class in classes:
            self.confusion_matrix_list[current_class] = AdaptedConfusionMatrix(current_class)

    def compute(self, nb_of_true_samples_for_classes):
        self.nb_of_true_samples_for_classes = nb_of_true_samples_for_classes
        sum_recall = 0
        recall = 0
        sum_soft_recall = 0
        weighted_soft_recall = 0
        sum_elements = 0
        for current_class in self.class_name:
            self.confusion_matrix_list[current_class].soft_recall()
            sum_recall += nb_of_true_samples_for_classes[current_class] * self.confusion_matrix_list[current_class].Recall
            recall += self.confusion_matrix_list[current_class].Recall
            sum_soft_recall += self.confusion_matrix_list[current_class].SoftRecall
            weighted_soft_recall += nb_of_true_samples_for_classes[current_class] * self.confusion_matrix_list[current_class].SoftRecall
            sum_elements += nb_of_true_samples_for_classes[current_class]

        self.SoftRecall = sum_soft_recall / len(self.class_name)
        self.WeightSoftRecall = weighted_soft_recall / sum_elements
        self.Recall = recall / len(self.class_name)
        self.WeightRecall = sum_recall / sum_elements

    def __str__(self):
        returned_string = 16 * " " + "TP    AN    FN   Precision   Recall    F1 score  SoftRecall\n"
        for current_class in self.class_name:
            returned_string += str(self.confusion_matrix_list[current_class]) + "\n"
        returned_string += "--\nSoftRecall\t\t\t" + str(self.SoftRecall)[:4]
        returned_string += "\nWeightedSoftRecall\t" + str(self.WeightSoftRecall)[:4]
        return returned_string


def soft_recall_function(ypred, ytrue, d, classes):
    matrix = acceptable_dict_to_matrix(d, classes)
    confusion_sof_recall = ConfusionSoftRecall(classes)

    nb_of_true_samples_for_classes = {}
    for current_class in classes:
        nb_of_true_samples_for_classes[current_class] = 0

    for index in range(ypred.shape[0]):
        predicted_label = get_class(ypred[index], classes)
        true_label = classes[ytrue[index].tolist().index(1.0)]

        nb_of_true_samples_for_classes[true_label] = nb_of_true_samples_for_classes[true_label] + 1

        if (predicted_label == true_label):
            confusion_sof_recall.confusion_matrix_list[true_label].TP += 1
        else:
            confusion_sof_recall.confusion_matrix_list[predicted_label].FP += 1
            if (matrix[true_label][predicted_label]):
                confusion_sof_recall.confusion_matrix_list[true_label].AN += 1
            else:
                confusion_sof_recall.confusion_matrix_list[true_label].FN += 1

    confusion_sof_recall.compute(nb_of_true_samples_for_classes)
    return confusion_sof_recall


def soft_recall_from_model(model, x_test, y_true, d, classes):
    y_pred = model.predict(x_test)
    return soft_recall_function(y_pred, y_true, d, classes)
