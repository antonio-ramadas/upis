#!/usr/bin/python3

import pandas as pd
import numpy as np
from sklearn.model_selection import ShuffleSplit
from Parser import Parser, DatasetPath, ActivityDataHeaders
from Headers import SensorProcessedDataHeaders


class DataProcessor:

    def __init__(self, data: pd.DataFrame=None, path=DatasetPath.MIT1):
        """
        Stores the data variable to an instance variable. If there is not one present (None is the default) then it
        parses the data from the path given (argument also optional).
        Lastly, the initialization does not process automatically the data which may be done by simply calling
        *process_sensors()*

        :param data: Pandas DataFrame with the parsed data
        :param path: Path to the file (dataset) to be read
        """
        if data is None:
            self.__data = Parser(path).data()
        else:
            self.__data = data

        self.path = path
        self.__data_processed = None

    @property
    def data_processed(self):
        if self.__data_processed is None:
            self.process_sensors()

        return self.__data_processed

    @data_processed.setter
    def data_processed(self, value):
        self.__data_processed = value

    def read(self, filename: str='sensors'):
        """
        Read the **processed data** from a csv file to a Pandas DataFrame.

        :param filename: Name of the file (exclude the csv extension)
        :return: Pandas DataFrame of the file read
        """
        file = 'processed/' + self.path.value + filename + '.csv'
        self.data_processed = pd.read_csv(file)

        if filename == 'sensors':
            sensor_id = SensorProcessedDataHeaders.ID
            activity  = SensorProcessedDataHeaders.ACTIVITY
            start     = SensorProcessedDataHeaders.START
            end       = SensorProcessedDataHeaders.END

            rename_columns = {
                sensor_id.value: sensor_id,
                activity.value:  activity
            }

            self.data_processed.rename(index=str, columns=rename_columns, inplace=True)

            self.data_processed[start] = pd.to_datetime(self.data_processed[start.value])
            self.data_processed[end]   = pd.to_datetime(self.data_processed[end.value])

            exclude_columns = {start.value, end.value}
            self.data_processed.drop(columns=exclude_columns, inplace=True)

        return self.data_processed

    def save(self, filename: str='sensors'):
        """
        Dump the processed data to a csv file.

        :param filename: Name of the file (without the csv extension)
        :param path: Path to the file (without the file name)
        """
        file = 'processed/' + self.path.value + filename + '.csv'
        self.data_processed.to_csv(file, index=False)

    def process_sensors(self):
        """
        It processes the original data and stores it to a instance variable. The original data is the one parsed from
        *activities_data.csv*. This method parses the original data where each row is an entry containing the sensor id,
        its activity and start and end timestamps. This information is stored on the variable instance *data_processed*
        which is also returned.

        :return: Processed data in a Pandas DataFrame structure
        """
        columns = [column for column in SensorProcessedDataHeaders]
        arr = np.empty((0,len(columns)))

        for _, row in self.__data.iterrows():
            # Get useful info
            ids           = row[ActivityDataHeaders.SENSOR_IDS]
            activity_name = row[ActivityDataHeaders.LABEL]
            start         = row[ActivityDataHeaders.SENSOR_ACTIVATION_TIMES]
            end           = row[ActivityDataHeaders.SENSOR_DEACTIVATION_TIMES]

            # Reshape to only 1 column
            ids           = np.array(ids).reshape((-1,1))
            activity_name = np.tile([[activity_name]], (ids.shape[0],1))
            start         = np.array(start).reshape((-1,1))
            end           = np.array(end).reshape((-1,1))

            # Stack side-by-side
            activity = np.hstack((ids,activity_name,start,end))

            # Stack bellow the data gathered so far
            arr = np.vstack((arr,activity))

        self.data_processed = pd.DataFrame(arr, columns=columns)

        return self.data_processed

    def split(self, n_folds=10, time_of_the_action = SensorProcessedDataHeaders.START):
        """
        Generate of splits following specific conditions. The parameter n_splits is the number of folds. By default it
        is 10-fold.
         - 66% to training data and the rest to test data.
        :return: Generator of Pandas DataFrames that split data into training and test set
        """
        # The data is spread across 2 months, but in the same year (look at the Jupyter Notebook)
        days_of_the_year = self.data_processed[time_of_the_action].apply(lambda x: x.dayofyear)
        days_of_the_year = days_of_the_year.to_frame()
        days_of_the_year.rename(index=str, columns={time_of_the_action:'day'}, inplace=True)

        hours = self.data_processed[time_of_the_action].apply(lambda x: x.hour)
        hours = hours.to_frame()
        hours.rename(index=str, columns={time_of_the_action:'hour'}, inplace=True)

        hours_and_days = pd.concat([hours, days_of_the_year], axis=1)

        cutoff = 5  # cutoff at 5am
        is_weekend = self.data_processed[time_of_the_action].apply(lambda x: x.weekday())
        is_weekend = is_weekend >= 5  # Saturday and Sunday are 5 and 6, respectively

        # Get weekdays and weekends as array of days of the year
        weekdays = days_of_the_year.loc[is_weekend.__array__() == False, 'day'].unique()
        weekends = days_of_the_year.loc[is_weekend.__array__(), 'day'].unique()

        test_size = 0.33

        # Split is made on the days and not on the rows
        # One shuffle is for weekdays and the other weekends. This way, we ensure that the weekends are also present
        ss_weekdays = ShuffleSplit(n_splits=n_folds, test_size=test_size)
        ss_weekends = ShuffleSplit(n_splits=n_folds, test_size=test_size)

        def __get_rows(days, split):
            """
            *split* is an array of indexes of the *days* and *days_of_the_year* is a column from the *data_processed* wh
            ich indicates the number of the day in the year that the action took place.
            :return: Pandas DataFrame with the actions that happened on the days indicated by the *split*
            """
            # Create function to map from the index to the element
            f = np.vectorize(lambda x: days[x])

            # Map it
            mapped = f(split)

            # Implement cutoff to separate the days (Jupyter Notebooks detail a bit this)
            # It is being created a new column to tell if the row took action during the weekend
            # If the action occurred before the cutoff, then it still counts to the previous day

            # Current day after *cutoff*
            lhs = hours_and_days['day'].isin(mapped).__array__() & (hours_and_days['hour'].__array__() >= cutoff)

            # Next day until *cutoff*
            rhs = hours_and_days['day'].isin(mapped+1).__array__() & (hours_and_days['hour'].__array__() < cutoff)

            return self.data_processed[lhs | rhs]

        # Iterate over the two generators at once
        for wdays, wend in zip(ss_weekdays.split(weekdays), ss_weekends.split(weekends)):
            wdays_train, wdays_test = wdays
            wend_train,  wend_test  = wend

            # Convert the indexes to days and get the actions occurred on that period
            wdays_train = __get_rows(weekdays, wdays_train)
            wdays_test  = __get_rows(weekdays, wdays_test)

            wend_train = __get_rows(weekends, wend_train)
            wend_test  = __get_rows(weekends, wend_test)

            yield wdays_train.append(wend_train), wdays_test.append(wend_test)

            #self.data_processed[days_of_the_year.isin(np.vectorize(lambda x: weekdays[x])(wdays_train))]


if __name__ == '__main__':
    print('Dataset processor')

    filename = 'sensors'
    path = DatasetPath.MIT2

    dp = DataProcessor(path=path)
    dp.save(filename)
    data = dp.read(filename)

    dp.split()
