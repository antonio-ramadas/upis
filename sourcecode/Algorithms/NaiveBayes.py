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

        col_id = SensorProcessedDataHeaders.ID
        col_start = SensorProcessedDataHeaders.START

        number_of_devices = self.__data[col_id].unique().size
        x = np.empty((0, number_of_devices))

        grouped = self.__data.groupby(by=SensorProcessedDataHeaders.ACTIVITY, sort=False)
        window_size = 15  # minutes

        for group_name, group in grouped:
            group.sort_values(by=col_start, ascending=False, inplace=True)

            # Difference between rows considering the start time
            group[col_start] = group[col_start].diff()

            first = True
            devices = {}

            for _, row in group.iterrows():
                if first:
                    devices |= row[col_id]
                    continue

                #if row[col_start] > window_size

                first = False
                devices |= row[col_id]


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

    #nb = NaiveBayes(data, NaiveBayesType.MULTIPLE)
    #nb.fit()

    sensor = 100
    #print('Prediction of the activity when sensor', sensor, 'is active:', nb.predict(sensor))
