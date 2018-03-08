#!/usr/bin/python3

from DataProcessor import DataProcessor
from Headers import SensorProcessedDataHeaders
from sklearn.ensemble import RandomForestClassifier
import pandas as pd


class RandomForest:

    def __init__(self, data: pd.DataFrame):
        self.__data = data
        self.__discretize_data()

        self.__rf = RandomForestClassifier()

    def __discretize_data(self):
        """
        Check Jupyter Notebooks for full explanation on the methods here applied.
        """
        pass

    def fit(self):
        activity_column = SensorProcessedDataHeaders.ACTIVITY

        x = self.__data.drop(columns=activity_column, inplace=True)
        y = self.__data[activity_column]

        self.__rf.fit(x, y)

    def predict(self):
        pass


if __name__ == '__main__':
    pass
    """
    print('Random Forest')

    dp = DataProcessor()
    data = dp.process_sensors()

    rf = RandomForest(data)
    rf.fit()

    print(rf.predict())
    """
