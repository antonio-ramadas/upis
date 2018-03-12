#!/usr/bin/python3

from DataProcessor import DataProcessor
from Parser import DatasetPath
from Headers import SensorProcessedDataHeaders
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import math

class RandomForest:

    def __init__(self, data: pd.DataFrame, dataset: DatasetPath):
        self.__dataset = dataset

        self.__data = self.__discretize_data(data)

        self.__rf = RandomForestClassifier()

    def __discretize_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Check Jupyter Notebooks for full explanation on the methods here applied.

        Basically, it creates four new columns (duration, duration categorized, weekday and period) and drops two (START
         and END).

        It returns the argument after being discretized.

        P.S.: May throw an exception if it is not implemented the categorization for the dataset given.
        """
        start = SensorProcessedDataHeaders.START
        end   = SensorProcessedDataHeaders.END

        # Add a new column containing the duration, in seconds, of each sensor action
        data = data.assign(duration=(data[end] - data[start]))
        data.loc[:, 'duration'] = data['duration'].apply(lambda x: x.total_seconds())

        # Categorize the duration of the actions
        if self.__dataset == DatasetPath.MIT1:
            data = data.assign(duration_categorized=pd.cut(data['duration'],
                                                           [-math.inf, 3, 11, 42, math.inf],
                                                           labels=False))
        elif self.__dataset == DatasetPath.MIT2:
            data = data.assign(duration_categorized=pd.cut(data['duration'],
                                                           [-math.inf, 5, 18, 232, math.inf],
                                                           labels=False))
        else:
            raise Exception('Dataset {} discretization not implemented'.format(self.__dataset))

        # Conversion to day of the week of the timestamp when the sensor activated
        data = data.assign(weekday=data[start].apply(lambda x: x.dayofweek))

        # Conversion of the start activity to the period of the day
        data = data.assign(period=pd.cut(data['duration'],
                                  [-math.inf, 6, 12, 17, math.inf],
                                  labels=False))

        data = data.drop(columns=[start, end])

        print(data.head())
        return data

    def fit(self):
        activity_column = SensorProcessedDataHeaders.ACTIVITY

        x = self.__data.drop(columns=activity_column)
        y = self.__data[activity_column]

        self.__rf.fit(x, y)

    def predict(self, test_data: pd.DataFrame):
        x = self.__discretize_data(test_data)

        x = x.drop(columns=[SensorProcessedDataHeaders.ACTIVITY])

        return self.__rf.predict(x)


if __name__ == '__main__':
    print('Random Forest')

    path = DatasetPath.MIT1

    dp = DataProcessor(path=path)
    data = dp.process_sensors()

    rf = RandomForest(data, path)
    rf.fit()

    row = dp.process_sensors().iloc[[0]]
    print(rf.predict(row))
