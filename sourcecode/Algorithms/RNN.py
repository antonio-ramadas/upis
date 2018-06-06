#!/usr/bin/python3
from random import random

from sklearn.preprocessing import LabelEncoder, StandardScaler, LabelBinarizer
from Headers import ActivityDataHeaders
from Metrics import Metrics
from Parser import Parser, DatasetPath
from DataProcessor import DataProcessor
from keras.layers.recurrent import LSTM, GRU
from keras.layers import Dense
from keras.models import Sequential
from keras.optimizers import RMSprop
from keras.callbacks import TensorBoard, EarlyStopping, History, Callback
from keras.models import load_model
import datetime
import time
import pandas as pd
import numpy as np
import keras.optimizers as optimizers
from keras import backend as K

class Resetter(Callback):
    def __init__(self, batches):
        super(Resetter, self).__init__()
        self.__sample_idx = 0
        self.__batch_idx = 0
        self.__batches = batches

    def on_batch_end(self, batch, logs=None):
        self.__sample_idx += 1
        if self.__sample_idx >= self.__batches[self.__batch_idx].shape[0]:
            self.__sample_idx = 0
            self.__batch_idx += 1
            self.__batch_idx %= len(self.__batches)

            self.model.reset_states()


def f1(y_true, y_pred):
    def recall(y_true, y_pred):
        """Recall metric.

        Only computes a batch-wise average of recall.

        Computes the recall, a metric for multi-label classification of
        how many relevant items are selected.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall

    def precision(y_true, y_pred):
        """Precision metric.

        Only computes a batch-wise average of precision.

        Computes the precision, a metric for multi-label classification of
        how many selected items are relevant.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision

    precision = precision(y_true, y_pred)
    recall = recall(y_true, y_pred)
    return 2 * ((precision * recall) / (precision + recall + K.epsilon()))

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
        #self.__scaler = StandardScaler()
        self.__hot_encoder = LabelBinarizer()

        # Encode the activities so they can act as index
        self.__encoder.fit(dp.data_processed[ActivityDataHeaders.LABEL].unique())

        self.__hot_encoder.fit(self.__encoder.transform(self.__encoder.classes_))

        #self.__hot_encoder.fit(dp.data_processed[ActivityDataHeaders.LABEL].unique())

        file_name = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S')

        file_name += " {}neu-{}lay-{}epo-{}-{}drop-{}lag-{}".format(self.__neurons,self.__n_layers,self.__n_epochs,
                                                                    self.__activation, self.__dropout, self.__lag,
                                                                    'LSTM' if self.__rnn_layer is LSTM else 'GRU')
        self.__file_name = file_name

    def __create_model(self):
        model = Sequential()

        input_shape = (self.__lag, 1)

        if self.__n_layers == 1:
            model.add(self.__rnn_layer(self.__neurons, input_shape=input_shape, stateful=True,
                                       batch_input_shape=(1,self.__lag,1),
                                       recurrent_dropout=self.__dropout, activation=self.__activation,
                                       recurrent_activation=self.__activation_r))
        else:
            model.add(self.__rnn_layer(self.__neurons, input_shape=input_shape, stateful=True,
                                       batch_input_shape=(1,self.__lag,1),
                                       recurrent_dropout=self.__dropout, activation=self.__activation,
                                       recurrent_activation=self.__activation_r, return_sequences=True))
            for i in range(1, self.__n_layers - 1):
                model.add(self.__rnn_layer(self.__neurons, recurrent_dropout=self.__dropout, stateful=True,
                                           activation=self.__activation, recurrent_activation=self.__activation_r,
                                           return_sequences=True))
            model.add(self.__rnn_layer(self.__neurons, recurrent_dropout=self.__dropout, stateful=True,
                                       activation=self.__activation, recurrent_activation=self.__activation_r))

        model.add(Dense(self.__dp.data_processed[ActivityDataHeaders.LABEL].unique().shape[0], activation='softmax'))

        # For now, let's not pass any further parameters
        model.compile(loss='categorical_crossentropy', optimizer=RMSprop(), metrics=[f1,'accuracy'])

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
        #starts[label] = self.__scaler.fit_transform(starts[label].values.reshape((-1, 1)))

        #ends = ends.rename(index=str, columns={ActivityDataHeaders.END_TIME: ActivityDataHeaders.START_TIME})

        #activities_df = starts.append(ends)
        activities_df = starts

        # By sorting, when an activity first shows, then it starts, and when it shows again, then it is its end
        activities_df.sort_values(ActivityDataHeaders.START_TIME, inplace=True)

        return activities_df

    def __create_batches(self, data:pd.DataFrame = None):
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

        y = self.__hot_encoder.transform(y)

        return x, y

    def load_model(self, file: str):
        self.__model = load_model("models/{}.h5".format(file))

    def save_model(self):
        self.__model.save("models/{}.h5".format(self.__file_name))

    def fit(self, data: pd.DataFrame = None, to_save=True) -> History:
        if data is None:
            data = self.__dp.data_processed

        batches = self.__create_batches(data)

        validation_size = 1 # number of days

        train_batches = batches[:-validation_size]
        validation_batches = batches[-validation_size:]

        train_x, train_y = self.__flat(train_batches)
        validation_x, validation_y = self.__flat(validation_batches)

        cbacks = []

        if to_save:
            cbacks.append(TensorBoard(log_dir="logs/{} {}".format(self.__file_name, random())))

        #cbacks.append(EarlyStopping(min_delta=1e-3, patience=25))
        cbacks.append(Resetter(train_batches))

        return self.__model.fit(x=train_x, y=train_y, validation_data=(validation_x,validation_y),
                         epochs=self.__n_epochs, shuffle=False, verbose=2, batch_size=1, callbacks=cbacks)

    def predict(self, x):
        batches = self.__create_batches(x)

        x, _ = self.__flat(batches)

        predictions = self.__model.predict(x, batch_size=1)

        predictions = self.__hot_encoder.inverse_transform(predictions)

        return predictions

        """

        #predictions = self.__scaler.inverse_transform(predictions)

        # Cast to int
        predictions = predictions.astype(int)

        n_classes = len(self.__encoder.classes_)

        new_predictions = np.empty(0)

        # This loop is because inverse_transform of Label Encoder throws an error if the number was never seen
        # https://github.com/scikit-learn/scikit-learn/issues/10552
        # This is just an workaround
        for idx in range(predictions.shape[0]):
            elem = predictions[idx]

            label = self.__encoder.inverse_transform(elem)[0] if 0 <= elem[0] < n_classes else ''

            new_predictions = np.hstack((new_predictions, label))

        return new_predictions
        """

    def evaluate(self, n_folds=10):
        matrices = []
        f1 = 0
        precision = 0
        recall = 0

        for train, test in self.__dp.split(n_folds, ActivityDataHeaders.START_TIME):
            self.fit(train, to_save=False)

            batches = self.__create_batches(test)
            _, truth = self.__flat(batches)
            truth = self.__hot_encoder.inverse_transform(truth)
            truth = pd.DataFrame(truth)

            prediction = self.predict(test)

            metric = Metrics(truth, pd.DataFrame(prediction))

            f1 += metric.f1()
            precision += metric.precision()
            recall += metric.recall()
            matrices += [metric.confusion_matrix()]

        f1 /= n_folds
        precision /= n_folds
        recall /= n_folds
        matrices = np.array(matrices)

        return f1, precision, recall, matrices


if __name__ == '__main__':
    print('Recurrent Neural Network')

    path = DatasetPath.MIT1

    dp = DataProcessor(path=path)

    dp.data_processed = Parser().data()

    rnn = RNN(dp, lag=5, neurons=64, n_layers=2, dropout=0, n_epochs=1, is_lstm=True)

    #rnn.fit()

    #predictions = rnn.predict(dp.data_processed)

    f1, precision, recall, matrices = rnn.evaluate()

    print(f'F1        = {f1}')
    print(f'Precision = {precision}')
    print(f'Recall    = {recall}')
