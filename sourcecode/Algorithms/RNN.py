#!/usr/bin/python3

from Parser import Parser, DatasetPath
from DataProcessor import DataProcessor
from keras.layers.recurrent import LSTM, GRU
from keras.layers import Dense
from keras.models import Sequential
from keras.optimizers import RMSprop
import pandas as pd


class RNN:
    def __init__(self, dp: DataProcessor, activation="tanh", activation_r="hard_sigmoid", lag=5, neurons=512,
                 dropout=0.2
                 , nlayers=1, is_lstm=True):
        self.__dp = dp
        self.__activation = activation
        self.__activation_r = activation_r
        self.__lag = lag
        self.__neurons = neurons
        self.__dropout = dropout
        self.__nlayers = nlayers
        self.__rnn_layer = LSTM if is_lstm else GRU
        self.__model = self.__create_model()
        self.__batches = []

    def __create_model(self):
        model = Sequential()

        if self.__nlayers == 1:
            model.add(self.__rnn_layer(self.__neurons, input_shape=(1, self.__lag),
                                       recurrent_dropout=self.__dropout, activation=self.__activation,
                                       recurrent_activation=self.__activation_r))
        else:
            model.add(self.__rnn_layer(self.__neurons, input_shape=(1, self.__lag),
                                       recurrent_dropout=self.__dropout, activation=self.__activation,
                                       recurrent_activation=self.__activation_r, return_sequences=True))
            for i in range(1, self.__nlayers - 1):
                model.add(self.__rnn_layer(self.__neurons, recurrent_dropout=self.__dropout,
                                           activation=self.__activation, recurrent_activation=self.__activation_r,
                                           return_sequences=True))
            model.add(self.__rnn_layer(self.__neurons, recurrent_dropout=self.__dropout,
                                       activation=self.__activation, recurrent_activation=self.__activation_r))

        model.add(Dense(1))

        # For now, let's not pass any further parameters
        model.compile(loss='mean_squared_error', optimizer=RMSprop(lr=0.001))

        return model

    def __make_batches(self, data:pd.DataFrame=None):
        if data is None:
            data = self.__dp.data_processed
        pass

    def fit(self, data: pd.DataFrame = None):
        if data is None:
            data = self.__dp.data_processed
        pass

    def predict(self, x):
        pass

    def evaluate(self, n_folds=10):
        return None, None, None, None


if __name__ == '__main__':
    print('Recurrent Neural Network')

    path = DatasetPath.MIT1

    dp = DataProcessor(path=path)

    dp.data_processed = Parser().data()

    rnn = RNN(dp)

    f1, precision, recall, matrices = rnn.evaluate()

    print(f'F1        = {f1}')
    print(f'Precision = {precision}')
    print(f'Recall    = {recall}')
