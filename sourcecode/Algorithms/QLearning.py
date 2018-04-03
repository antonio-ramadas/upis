#!/usr/bin/python3

from DataProcessor import DataProcessor
from Parser import DatasetPath
import pandas as pd

"""
Algorithm from:
Reaz, M. B. I., Assim, A., Ibrahimy, M. I., Choong, F., & Mohd-Yasin, F. (2008).
Smart Home Device Usage Prediction Using Pattern Matching and Reinforcement Learning.
7th International Conference on System Identification and Control Problems (SCIPRO’08), (February 2016), 1092–1098.
Retrieved from http://www.econf.info/files/105/1345/1092.pdf
"""


class QLearning:
    def __init__(self, dp: DataProcessor):
        self.__dp = dp

    def fit(self, train: pd.DataFrame):
        # process data set

        # steps 1..7

        pass


if __name__ == '__main__':
    print('Q-Learning')

    path = DatasetPath.MIT1

    dp = DataProcessor(path=path)

    ql = QLearning(dp)
    ql.fit(dp.data_processed)