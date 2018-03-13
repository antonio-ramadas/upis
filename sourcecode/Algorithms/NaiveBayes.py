#!/usr/bin/python3

from Parser import DatasetPath
from DataProcessor import DataProcessor
from Headers import SensorProcessedDataHeaders, NaiveBayesType
from sklearn.preprocessing import LabelBinarizer
from sklearn.naive_bayes import GaussianNB
from Metrics import *
import numpy as np
import pandas as pd


class NaiveBayes:

    def __init__(self, dp: DataProcessor, type=NaiveBayesType.SINGLE):
        self.__dp      = dp
        self.__type    = type
        # __encoder is currently only being used in MULTIPLE
        self.__encoder = LabelBinarizer()
        self.__nb      = GaussianNB()

    def __add_devices_with_encoder(self, x, devices):
        """
        The idea here is simple, but may be complex if the workflow is not completely followed.
        The transform of the encoder returns a matrix where each row is one hot vector
            (one element is 1 and the rest zeros
                (note: assuming that there are no repetitive
                       elements given in the devices variable)
            )
        Given that we want only 1 row, then we sum all columns
        """
        devices = list(set(devices))
        return np.vstack((x, np.sum(self.__encoder.transform(devices), axis=0)))

    # window_size in minutes
    def __encode_input_of_multiple(self, data, window_size=15):
        col_id = SensorProcessedDataHeaders.ID
        col_start = SensorProcessedDataHeaders.START

        # x is the training data
        x = np.empty((0, self.__number_of_devices))
        y = np.empty((0, 1))

        grouped = data.groupby(by=SensorProcessedDataHeaders.ACTIVITY, sort=False)
        window_size *= 60  # convert minutes to seconds

        for group_name, group in grouped:
            # This produces a warning due to inplace. It is safe to ignore it.
            group.sort_values(by=col_start, ascending=True, inplace=True)

            # Difference between rows considering the start time
            # This produces a warning due to the assignment. It is safe to ignore it.
            group[col_start] = group[col_start].diff()

            # List of devices acting together
            devices = set()

            for _, row in group.iterrows():
                # If it is not the first (diff is NaT) and the difference is bigger than the window
                if not pd.isnull(row[col_start]) and row[col_start].total_seconds() > window_size:
                    x = self.__add_devices_with_encoder(x, devices)
                    y = np.vstack((y, [group_name]))
                    devices = set()

                devices |= {row[col_id]}

            x = self.__add_devices_with_encoder(x, devices)
            y = np.vstack((y, [group_name]))

        return x, y

    def __fit_multiple(self, data):
        col_id = SensorProcessedDataHeaders.ID

        # Fit the encoder to all known sensors
        self.__encoder.fit(data[col_id].unique())

        self.__number_of_devices = data[col_id].unique().size

        x, y = self.__encode_input_of_multiple(data)
        self.__nb.fit(x, y)

    def __fit_single(self, data):
        x = data[SensorProcessedDataHeaders.ID].values.reshape(-1, 1)
        y = data[SensorProcessedDataHeaders.ACTIVITY]
        self.__nb.fit(x, y)

    def fit(self, data):
        if self.__type is NaiveBayesType.SINGLE:
            self.__fit_single(data)
        elif self.__type is NaiveBayesType.MULTIPLE:
            self.__fit_multiple(data)

    def predict(self, sensor):
        x = None

        if self.__type is NaiveBayesType.SINGLE:
            x = sensor[SensorProcessedDataHeaders.ID].values.reshape(-1, 1)
        elif self.__type is NaiveBayesType.MULTIPLE:
            x, _ = self.__encode_input_of_multiple(sensor)

        return self.__nb.predict(x)

    def evaluate(self, n_folds=10):
        matrices  = []
        f1        = 0
        precision = 0
        recall    = 0

        for train, test in self.__dp.split(n_folds=n_folds):
            self.fit(train)

            truth = None

            if self.__type is NaiveBayesType.SINGLE:
                truth = test[SensorProcessedDataHeaders.ACTIVITY]
            elif self.__type is NaiveBayesType.MULTIPLE:
                _, truth = self.__encode_input_of_multiple(test)

            prediction = self.predict(test)

            metric = Metrics(truth, prediction)

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
    print('Naive Bayes')

    filename = 'sensors'
    path = DatasetPath.MIT1

    dp = DataProcessor(path=path)

    nb = NaiveBayes(dp, NaiveBayesType.SINGLE)
    nb.fit(dp.data_processed)

    print(nb.predict(dp.process_sensors().iloc[[0]]))

    #sensor = ['100', '101', '95', '54', '93','72','67','108']
    #print('Prediction of the activity when sensor', sensor, 'is active:', nb.predict(sensor))

    nb.evaluate()
