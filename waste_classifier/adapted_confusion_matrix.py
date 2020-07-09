class AdaptedConfusionMatrix:
    def __init__(self, name: str, mode: str = "adaptive"):
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

