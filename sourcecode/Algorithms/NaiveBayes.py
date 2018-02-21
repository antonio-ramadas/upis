#!/usr/bin/python3

from Parser import DatasetPath
from DataProcessor import DataProcessor
from Headers import SensorProcessedDataHeaders
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import GaussianNB


class NaiveBayes:

    def __init__(self, data):
        self.__data = data
        self.__encoder = LabelEncoder()
        self.__nb = GaussianNB()

    def fit(self):
        x = self.__data[SensorProcessedDataHeaders.ID].values.reshape(-1, 1)

        activities = self.__data[SensorProcessedDataHeaders.ACTIVITY]
        y = self.__encoder.fit_transform(activities)

        self.__nb.fit(x, y)

    def predict(self, sensor_id):
        return self.__encoder.inverse_transform(self.__nb.predict(sensor_id))


if __name__ == '__main__':
    print('Naive Bayes')

    filename = 'sensors'
    path = DatasetPath.MIT1

    dp = DataProcessor(path=path)
    dp.process_sensors()
    data = dp.data_processed

    nb = NaiveBayes(data)
    nb.fit()

    sensor = 100
    print('Prediction of the activity when sensor', sensor, 'is active:', nb.predict(sensor))
