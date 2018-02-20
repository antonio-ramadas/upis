#!/usr/bin/python3

import pandas as pd

from itertools import islice
from enum import Enum
from Headers import ActivityDataHeaders


class DatasetPath(Enum):
    MIT1 = 'MIT/subject1/'
    MIT2 = 'MIT/subject2/'


class Parser:
    __DATASET_PATH = ''

    def __init__(self, ds = DatasetPath.MIT1):
        self.__DATASET_PATH = 'datasets/' + ds.value

    def __read_file(self, filename):
        return pd.read_csv(self.__DATASET_PATH + filename)
    
    def __read_n_lines(self, file, n=5):
        return [line.strip() for line in list(islice(file, n))]

    def sensors(self):
        return self.__read_file('sensors.csv')

    def activities(self):
        return self.__read_file('activities.csv')

    def data(self):
        df = pd.DataFrame(columns=[ActivityDataHeaders.LABEL,
                                   ActivityDataHeaders.DATE,
                                   ActivityDataHeaders.START_TIME,
                                   ActivityDataHeaders.END_TIME,
                                   ActivityDataHeaders.SENSOR_IDS,
                                   ActivityDataHeaders.SENSOR_OBJECTS,
                                   ActivityDataHeaders.SENSOR_ACTIVATION_TIMES,
                                   ActivityDataHeaders.SENSOR_DEACTIVATION_TIMES])

        delimiter = ','

        with open(self.__DATASET_PATH + 'activities_data.csv') as f:
            # An activity is defined on 5 lines
            n=5
            rows = []
            lines = self.__read_n_lines(f, n=n)

            while lines:
                info                = lines[0].split(delimiter)
                date                = info[1] + ' '
                sensor_ids          = lines[1].split(delimiter)
                sensor_objects      = lines[2].split(delimiter)
                sensor_activation   = [pd.Timestamp(date + time) for time in lines[3].split(delimiter)]
                sensor_deactivation = [pd.Timestamp(date + time) for time in lines[4].split(delimiter)]

                rows += [{
                            ActivityDataHeaders.LABEL                     : info[0],
                            ActivityDataHeaders.DATE                      : pd.Timestamp(info[1]),
                            ActivityDataHeaders.START_TIME                : pd.Timestamp(date + info[2]),
                            ActivityDataHeaders.END_TIME                  : pd.Timestamp(date + info[3]),
                            ActivityDataHeaders.SENSOR_IDS                : sensor_ids,
                            ActivityDataHeaders.SENSOR_OBJECTS            : sensor_objects,
                            ActivityDataHeaders.SENSOR_ACTIVATION_TIMES   : sensor_activation,
                            ActivityDataHeaders.SENSOR_DEACTIVATION_TIMES : sensor_deactivation
                        }]

                lines = self.__read_n_lines(f, n=n)

        # According to Python Documentation, it is faster to append a chunk than small pieces
        df = df.append(rows)
        
        return df


if __name__ == '__main__':
    print('Read MIT dataset')

    sensors = Parser().sensors()
    print(sensors.head())

    activities = Parser().activities()
    print(activities.head())

    data = Parser().data()
    print(data.head())
