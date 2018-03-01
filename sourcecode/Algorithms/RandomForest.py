#!/usr/bin/python3

from DataProcessor import DataProcessor
import pandas as pd


class RandomForest:

    def __init__(self, data:pd.DataFrame):
        self.__data__ = data

    def fit(self):
        pass

    def predict(self):
        pass


if __name__ == '__main__':
    print('Random Forest')

    dp = DataProcessor()
    data = dp.process_sensors()

    rf = RandomForest(data)
    rf.fit()

    print(rf.predict())