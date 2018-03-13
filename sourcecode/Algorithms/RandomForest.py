#!/usr/bin/python3
from DataProcessor import DataProcessor
from Parser import DatasetPath
from Headers import SensorProcessedDataHeaders
from sklearn.ensemble import RandomForestClassifier
from Metrics import *
import pandas as pd
import numpy as np
import math


class RandomForest:
    def __init__(self, dp:DataProcessor):
        self.__dp = dp

        self.__rf = RandomForestClassifier()

    def __discretize_data(self, old_data: pd.DataFrame) -> pd.DataFrame:
        """
        Check Jupyter Notebooks for full explanation on the methods here applied.

        Basically, it creates four new columns (duration, duration categorized, weekday and period) and drops two (START
         and END).

        It returns the argument after being discretized.

        P.S.: May throw an exception if it is not implemented the categorization for the dataset given.
        """
        data = old_data.copy()

        start = SensorProcessedDataHeaders.START
        end   = SensorProcessedDataHeaders.END

        # Add a new column containing the duration, in seconds, of each sensor action
        data = data.assign(duration=(data[end] - data[start]))
        data.loc[:, 'duration'] = data['duration'].apply(lambda x: x.total_seconds())

        # Categorize the duration of the actions
        if self.__dp.path == DatasetPath.MIT1:
            data = data.assign(duration_categorized=pd.cut(data['duration'],
                                                           [-math.inf, 3, 11, 42, math.inf],
                                                           labels=False))
        elif self.__dp.path == DatasetPath.MIT2:
            data = data.assign(duration_categorized=pd.cut(data['duration'],
                                                           [-math.inf, 5, 18, 232, math.inf],
                                                           labels=False))
        else:
            raise Exception('Dataset {} discretization not implemented'.format(self.__dp.path))

        # Conversion to day of the week of the timestamp when the sensor activated
        data = data.assign(weekday=data[start].apply(lambda x: x.dayofweek))

        # Conversion of the start activity to the period of the day
        data = data.assign(period=pd.cut(data['duration'],
                                  [-math.inf, 6, 12, 17, math.inf],
                                  labels=False))

        data = data.drop(columns=[start, end])

        return data

    def fit(self, train: pd.DataFrame):
        activity_column = SensorProcessedDataHeaders.ACTIVITY

        train = self.__discretize_data(train)

        x = train.drop(columns=activity_column)
        y = train[activity_column]

        self.__rf.fit(x, y)

    def predict(self, test: pd.DataFrame):
        x = self.__discretize_data(test)

        x = x.drop(columns=[SensorProcessedDataHeaders.ACTIVITY])

        return self.__rf.predict(x)

    def evaluate(self, n_folds=10):
        activity_column = SensorProcessedDataHeaders.ACTIVITY
        matrices  = []
        f1        = 0
        precision = 0
        recall    = 0

        for train, test in self.__dp.split(n_folds=n_folds):
            self.fit(train)
            prediction = self.predict(test)

            metric = Metrics(test[activity_column], prediction)

            f1        += metric.f1()
            precision += metric.precision()
            recall    += metric.recall()
            matrices  += [metric.confusion_matrix()]

        f1        /= n_folds
        precision /= n_folds
        recall    /= n_folds
        matrices   = np.array(matrices)

        return f1, precision, recall, matrices


if __name__ == '__main__':
    print('Random Forest')

    path = DatasetPath.MIT1

    dp = DataProcessor(path=path)

    rf = RandomForest(dp)
    rf.fit(dp.data_processed)

    row = dp.process_sensors().iloc[[0]]
    print(rf.predict(row))

    f1, precision, recall, matrices = rf.evaluate()

    print(f'F1        = {f1}')
    print(f'Precision = {precision}')
    print(f'Recall    = {recall}')