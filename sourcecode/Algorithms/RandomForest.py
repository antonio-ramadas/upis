#!/usr/bin/python3

from DataProcessor import DataProcessor
from Parser import DatasetPath
from Headers import SensorProcessedDataHeaders
from sklearn.ensemble import RandomForestClassifier
import pandas as pd


class RandomForest:

    def __init__(self, data: pd.DataFrame, dataset: DatasetPath):
        self.__dataset = dataset

        self.__data = data
        self.__discretize_data()

        self.__rf = RandomForestClassifier()

    def __discretize_data(self):
        """
        Check Jupyter Notebooks for full explanation on the methods here applied.
        """
        start = SensorProcessedDataHeaders.START
        end   = SensorProcessedDataHeaders.END

        # Add a new column containing the duration, in seconds, of each sensor action
        self.__data['duration'] = self.__data[end] - self.__data[start]
        self.__data['duration'] = self.__data['duration'].apply(lambda x: x.total_seconds())

        # Categorize the duration of the actions
        # TODO
        self.__data['duration_categorized'] = self.__data['duration']

        if self.__dataset == DatasetPath.MIT1:
            pass
        elif self.__dataset == DatasetPath.MIT2:
            pass
        else:
            raise Exception('Dataset {} discretization not implemented'.format(self.__dataset))

        # Conversion to day of the week of the timestamp when the sensor activated
        self.__data['weekday'] = self.__data[start].apply(lambda x: x.weekday_name)

        # Conversion of the start activity to the period of the day
        # TODO

    def fit(self):
        activity_column = SensorProcessedDataHeaders.ACTIVITY

        x = self.__data.drop(columns=activity_column, inplace=True)
        y = self.__data[activity_column]

        self.__rf.fit(x, y)

    def predict(self):
        pass


if __name__ == '__main__':
    print('Random Forest')

    path = DatasetPath.MIT1

    dp = DataProcessor(path=path)
    data = dp.process_sensors()

    rf = RandomForest(data, path)
    """
    rf.fit()

    print(rf.predict())
    """
