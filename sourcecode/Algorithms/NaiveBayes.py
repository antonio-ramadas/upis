#!/usr/bin/python3

from Parser import DatasetPath
from DataProcessor import DataProcessor
from sklearn.naive_bayes import GaussianNB

class NaiveBayes:
    __data = None

    def __init__(self, data):
        self.__data = data

    def execute(self):
        pass

if __name__ == '__main__':
    print('Naive Bayes')

    filename = 'sensors'
    path = DatasetPath.MIT1

    dp = DataProcessor(path=path)
    dp.process_sensors()
    data = dp.data_processed

    nb = NaiveBayes(data)
    nb.execute()
