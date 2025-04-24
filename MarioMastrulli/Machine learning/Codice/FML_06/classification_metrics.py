import numpy as np


class ClassificationMetrics:
    def __init__(self, model):
        self.model = model
        self.cm = None
        self.tn = None
        self.fp = None
        self.fn = None
        self.tp = None

    def compute_performance(self, X, y):
        preds = self.model.predict(X).squeeze()
        self.cm = self.confusion_matrix(preds, y)
        self.tn = self.cm[0, 0]
        self.fp = self.cm[0, 1]
        self.fn = self.cm[1, 0]
        self.tp = self.cm[1, 1]

        return {
            "confusion_matrix": self.cm,
            "accuracy": self.accuracy(),
            "error_rate": self.error_rate(),
            "precision": self.precision(),
            "recall": self.recall(),
            "fn_rate": self.fn_rate(),
            "specificity": self.specificity(),
            "fp_rate": self.fp_rate(),
            "f1_score": self.f1_score(),
        }

    def confusion_matrix(self, preds, y):
        m = len(y)
        tp, tn, fp, fn = 0, 0, 0, 0

        for i in range(m):
            if preds[i] == y[i]:
                if preds[i] == 0:
                    tn += 1
                else:
                    tp += 1
            else:
                if preds[i] == 0:
                    fn += 1
                else:
                    fp += 1

        return np.array([[tn, fp], [fn, tp]])

    def accuracy(self):
        return (self.tp + self.tn) / self.cm.sum()

    def error_rate(self):
        return 1 - self.accuracy()

    def precision(self):
        return self.tp / (self.tp + self.fp)

    def recall(self):
        return self.tp / (self.tp + self.fn)

    def fn_rate(self):
        return 1 - self.recall()

    def specificity(self):
        return self.tn / (self.tn + self.fp)

    def fp_rate(self):
        return 1 - self.specificity()

    def f1_score(self):
        precision = self.precision()
        recall = self.recall()

        return 2 * (precision * recall) / (precision + recall)
