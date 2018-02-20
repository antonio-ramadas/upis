#!/usr/bin/python3

import pandas as pd
import numpy as np
from Parser import Parser, DatasetPath, ActivityDataHeaders
from Headers import SensorProcessedDataHeaders


class DataProcessor:
    __data = None
    __path = None
    data_processed = None

    def __init__(self, data=None, path=DatasetPath.MIT1):
        if (data is None):
            self.__data = Parser(path).data()
        else:
            self.__data = data

        self.__path = path
    
    def read(self, filename='sensors', path=DatasetPath.MIT1):
        file = 'processed/' + path.value + filename + '.csv'
        self.data_processed = pd.read_csv(file)

        if (filename == 'sensors'):
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

    def save(self, filename='sensors', path=DatasetPath.MIT1):
        file = 'processed/' + path.value + filename + '.csv'
        self.data_processed.to_csv(file, index=False)

    def process_sensors(self):
        columns = [column for column in SensorProcessedDataHeaders]
        arr = np.empty((0,len(columns)))

        for _, row in self.__data.iterrows():
            # Get useful info
            ids   = row[ActivityDataHeaders.SENSOR_IDS]
            activity_name = row[ActivityDataHeaders.LABEL]
            start = row[ActivityDataHeaders.SENSOR_ACTIVATION_TIMES]
            end   = row[ActivityDataHeaders.SENSOR_DEACTIVATION_TIMES]

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


if __name__ == '__main__':
    print('Dataset processor')

    filename = 'sensors'
    path = DatasetPath.MIT1

    dp = DataProcessor(path=path)
    dp.process_sensors()
    dp.save(filename, path)
    data = dp.read(filename, path)