#!/usr/bin/python3

import sklearn.metrics.classification as skmetrics
import pandas as pd


class Metrics:

    def __init__(self, truth: pd.DataFrame, prediction: pd.DataFrame):
        self.__ground_truth = truth
        self.__prediction = prediction

    def __apply(self, f):
        return f(self.__ground_truth, self.__prediction, average='macro')

    def f1(self):
        return self.__apply(skmetrics.f1_score)

    def precision(self):
        return self.__apply(skmetrics.precision_score)

    def recall(self):
        return self.__apply(skmetrics.recall_score)

    def confusion_matrix(self):
        return skmetrics.confusion_matrix(self.__ground_truth, self.__prediction)


if __name__ == '__main__':
    print('Metrics')

    gt = pd.DataFrame([i for i in range(0, 10)])
    pc = pd.DataFrame([i for i in range(0, 5, 1)]).append([i for i in range(15, 20)])

    mt = Metrics(gt, pc)

    print(mt.f1())
    print(mt.confusion_matrix())
    print(mt.precision())
    print(mt.recall())
