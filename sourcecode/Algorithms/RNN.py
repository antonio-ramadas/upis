#!/usr/bin/python3
from sklearn.preprocessing import LabelEncoder
from Headers import ActivityDataHeaders
from Parser import Parser, DatasetPath
from DataProcessor import DataProcessor
from keras.layers.recurrent import LSTM, GRU
from keras.layers import Dense
from keras.models import Sequential
from keras.optimizers import RMSprop
from keras.callbacks import TensorBoard, EarlyStopping
import datetime
import time
import pandas as pd
import numpy as np

class RNN:
    def __init__(self, dp: DataProcessor, activation="tanh", activation_r="hard_sigmoid", lag=5, neurons=512,
                 dropout=0.2, n_layers=1, n_epochs=500, is_lstm=True):
        self.__dp = dp
        self.__activation = activation
        self.__activation_r = activation_r
        self.__lag = lag
        self.__neurons = neurons
        self.__dropout = dropout
        self.__n_layers = n_layers
        self.__n_epochs = n_epochs
        self.__rnn_layer = LSTM if is_lstm else GRU
        self.__model = self.__create_model()
        self.__cutoff = 5 # 5 am
        self.__encoder = LabelEncoder()

        # Encode the activities so they can act as index
        self.__encoder.fit(dp.data_processed[ActivityDataHeaders.LABEL].unique())

    def __create_model(self):
        model = Sequential()

        input_shape = (self.__lag, 1)

        if self.__n_layers == 1:
            model.add(self.__rnn_layer(self.__neurons, input_shape=input_shape,
                                       recurrent_dropout=self.__dropout, activation=self.__activation,
                                       recurrent_activation=self.__activation_r))
        else:
            model.add(self.__rnn_layer(self.__neurons, input_shape=input_shape,
                                       recurrent_dropout=self.__dropout, activation=self.__activation,
                                       recurrent_activation=self.__activation_r, return_sequences=True))
            for i in range(1, self.__n_layers - 1):
                model.add(self.__rnn_layer(self.__neurons, recurrent_dropout=self.__dropout,
                                           activation=self.__activation, recurrent_activation=self.__activation_r,
                                           return_sequences=True))
            model.add(self.__rnn_layer(self.__neurons, recurrent_dropout=self.__dropout,
                                       activation=self.__activation, recurrent_activation=self.__activation_r))

        model.add(Dense(1))

        # For now, let's not pass any further parameters
        model.compile(loss='mean_squared_error', optimizer=RMSprop(lr=0.001))

        return model

    # TODO this is the same method as in QLearning, but with small differences
    def __process_dataset(self, activities_df) -> pd.DataFrame:
        """
        Process the dataset into 2 columns: Label and Time it occurred. The dataset as given has 3 columns (Label, Start
        and End). The return is the dataset sorted by time. So the length of an activity is the time between two
        consecutive occurrences. When an activity first shows, then it started. When it shows again, then it ended. Just
        loop between these two states while you traverse the dataset and it is possible to point out the exact state of
        the entire house in any point of time.

        :return: Sorted dataset, by time, with two columns (Label and Time)
        """
        label = ActivityDataHeaders.LABEL

        starts = activities_df[[label, ActivityDataHeaders.START_TIME]]
        #ends = activities_df[[label, ActivityDataHeaders.END_TIME]]

        starts[label] = starts[label].apply(lambda x: self.__encoder.transform([x])[0])

        #ends = ends.rename(index=str, columns={ActivityDataHeaders.END_TIME: ActivityDataHeaders.START_TIME})

        #activities_df = starts.append(ends)
        activities_df = starts

        # By sorting, when an activity first shows, then it starts, and when it shows again, then it is its end
        activities_df.sort_values(ActivityDataHeaders.START_TIME, inplace=True)

        return activities_df

    def __create_batches(self, data:pd.DataFrame):
        if data is None:
            data = self.__dp.data_processed

        activities_df = self.__process_dataset(data)

        batches = []
        batch = np.empty((0,0))
        day = -1

        for _, row in activities_df.iterrows():
            row_day = row[ActivityDataHeaders.START_TIME].dayofyear
            row_hour = row[ActivityDataHeaders.START_TIME].hour

            if day == -1:
                batch = np.empty((0,0))
                day = row_day

            if (day == row_day and row_hour >= self.__cutoff) or (day == row_day+1 and row_hour < self.__cutoff):
                batch = np.append(batch, row[ActivityDataHeaders.LABEL])
            else:
                batches.append(batch)
                batch = np.empty((0,0))
                day = row_day

        for idx, batch in enumerate(batches):
            new_batch = np.empty((0, self.__lag))
            y = batch[self.__lag:].reshape((-1,1))

            for activity_idx in range(self.__lag, batch.shape[0]):
                new_batch = np.vstack((new_batch, batch[activity_idx-self.__lag:activity_idx]))

            batches[idx] = np.hstack((new_batch,y))

        return batches

    def __flat(self, batches):
        x = np.empty((0,self.__lag))
        y = np.empty((0,1))

        for batch in batches:
            x = np.vstack((x,batch[:,:-1]))
            y = np.vstack((y,batch[:,-1:]))

        x = x.reshape((x.shape[0],self.__lag,1))

        return x, y

    def fit(self, data: pd.DataFrame = None):
        if data is None:
            data = self.__dp.data_processed

        batches = self.__create_batches(data)

        validation_size = 0.3
        validation_size = int(len(batches) * validation_size)

        train_batches = batches[:-validation_size]
        validation_batches = batches[-validation_size:]

        train_x, train_y = self.__flat(train_batches)
        validation_x, validation_y = self.__flat(validation_batches)

        file_time = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S')

        file_time += " {}neu-{}lay-{}epo-{}-{}drop-{}lag-{}".format(self.__neurons,self.__n_layers,self.__n_epochs,
                                                                       self.__activation, self.__dropout, self.__lag,
                                                                       'LSTM' if self.__rnn_layer is LSTM else 'GRU')

        tensorboard = TensorBoard(log_dir="logs/{}".format(file_time))

        eayly_stopping = EarlyStopping(min_delta=10**-10, patience=25)

        self.__model.fit(x=train_x, y=train_y, validation_data=(validation_x,validation_y), epochs=self.__n_epochs,
                         shuffle=False, verbose=2, batch_size=None, callbacks=[tensorboard, eayly_stopping])

    def predict(self, x):
        pass

    def evaluate(self, n_folds=10):
        return None, None, None, None


if __name__ == '__main__':
    print('Recurrent Neural Network')

    path = DatasetPath.MIT1

    dp = DataProcessor(path=path)

    dp.data_processed = Parser().data()

    rnn = RNN(dp, neurons=16, n_layers=3, dropout=0.5)

    rnn.fit()

    f1, precision, recall, matrices = rnn.evaluate()

    print(f'F1        = {f1}')
    print(f'Precision = {precision}')
    print(f'Recall    = {recall}')
