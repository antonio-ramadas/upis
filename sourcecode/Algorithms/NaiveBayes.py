#!/usr/bin/python3

from Parser import DatasetPath
from DataProcessor import DataProcessor
from Headers import SensorProcessedDataHeaders, NaiveBayesType
from sklearn.preprocessing import LabelBinarizer
from sklearn.naive_bayes import GaussianNB
import numpy as np


class NaiveBayes:

    def __init__(self, data, type=NaiveBayesType.SINGLE):
        self.__data    = data
        self.__type    = type
        self.__encoder = LabelBinarizer()
        self.__nb      = GaussianNB()

    def __fit_multiple(self):
        print('Not working yet')
        """
        self.__encoder.fit(self.__data[SensorProcessedDataHeaders.ID].unique())
        x = self.__encoder.transform(self.__data[SensorProcessedDataHeaders.ID])
        x = np.sum(x, axis=0)
        print(x)

        y = self.__data[SensorProcessedDataHeaders.ACTIVITY]

        self.__nb.fit(x, y)
        """

    def __fit_single(self):
        x = self.__data[SensorProcessedDataHeaders.ID].values.reshape(-1, 1)
        y = self.__data[SensorProcessedDataHeaders.ACTIVITY]
        self.__nb.fit(x, y)

    def fit(self):
        if self.__type is NaiveBayesType.SINGLE:
            self.__fit_single()
        elif self.__type is NaiveBayesType.MULTIPLE:
            self.__fit_multiple()

    def predict(self, sensor_id):
        return self.__nb.predict(sensor_id)
        #return self.__encoder.inverse_transform(self.__nb.predict(sensor_id))


if __name__ == '__main__':
    print('Naive Bayes')

    filename = 'sensors'
    path = DatasetPath.MIT1

    dp = DataProcessor(path=path)
    dp.process_sensors()
    data = dp.data_processed

    nb = NaiveBayes(data, NaiveBayesType.MULTIPLE)
    nb.fit()

    sensor = 100
    #print('Prediction of the activity when sensor', sensor, 'is active:', nb.predict(sensor))
