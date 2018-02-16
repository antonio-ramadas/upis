#!/usr/bin/python3

import pandas as pd

from itertools import islice
from enum import Enum

class ActivityDataHeaders(Enum):
    LABEL                     = 'ACTIVITY_LABEL',
    DATE                      = 'DATE',
    START_TIME                = 'START_TIME',
    END_TIME                  = 'END_TIME',
    SENSOR_IDS                = 'SENSOR_IDS',
    SENSOR_OBJECTS            = 'SENSOR_OBJECTS',
    SENSOR_ACTIVATION_TIMES   = 'SENSOR_ACTIVATION_TIMES',
    SENSOR_DEACTIVATION_TIMES = 'SENSOR_DEACTIVATION_TIMES'

class Parser:
    DATASET_PATH = 'datasets/MIT/subject1/'

    def __read_file(self, filename):
        return pd.read_csv(self.DATASET_PATH + filename)

    def sensors(self):
        return self.__read_file('sensors.csv')

    def activities(self):
        return self.__read_file('activities.csv')

    def data(self):
        df = pd.DataFrame(columns=[ActivityDataHeaders.LABEL,                     \
                                   ActivityDataHeaders.DATE,                      \
                                   ActivityDataHeaders.START_TIME,                \
                                   ActivityDataHeaders.END_TIME,                  \
                                   ActivityDataHeaders.SENSOR_IDS,                \
                                   ActivityDataHeaders.SENSOR_OBJECTS,            \
                                   ActivityDataHeaders.SENSOR_ACTIVATION_TIMES,   \
                                   ActivityDataHeaders.SENSOR_DEACTIVATION_TIMES  ])
        
        delimiter = ','

        with open(self.DATASET_PATH + 'activities_data.csv') as f:
            # TODO watch out for activities that start on a day and end on the next day

            # An activity is spread accross 5 lines
            # TODO repetitive code
            lines = [line.strip() for line in list(islice(f, 5))]

            while lines:
                info                = lines[0].split(delimiter)
                date                = info[1] + ' '
                sensor_ids          = lines[1].split(delimiter)
                sensor_objects      = lines[2].split(delimiter)
                sensor_activation   = [pd.Timestamp(date + time) for time in lines[3].split(delimiter)]
                sensor_deactivation = [pd.Timestamp(date + time) for time in lines[4].split(delimiter)]

                # TODO from PyDocs -> Iteratively appending rows to a DataFrame can be more computationally
                # intensive than a single concatenate. A better solution is to append those rows to a list and
                # then concatenate the list with the original DataFrame all at once.
                df = df.append({
                    ActivityDataHeaders.LABEL                     : info[0],
                    ActivityDataHeaders.DATE                      : pd.Timestamp(info[1]),
                    ActivityDataHeaders.START_TIME                : pd.Timestamp(date + info[2]),
                    ActivityDataHeaders.END_TIME                  : pd.Timestamp(date + info[3]),
                    ActivityDataHeaders.SENSOR_IDS                : sensor_ids,
                    ActivityDataHeaders.SENSOR_OBJECTS            : sensor_objects,
                    ActivityDataHeaders.SENSOR_ACTIVATION_TIMES   : sensor_activation,
                    ActivityDataHeaders.SENSOR_DEACTIVATION_TIMES : sensor_deactivation
                }, ignore_index=True)

                lines = [line.strip() for line in list(islice(f, 5))]
        
        return df


if __name__ == '__main__':
    print('Read MIT dataset')

    sensors = Parser().sensors()
    print(sensors.head())

    activities = Parser().activities()
    print(activities.head())

    data = Parser().data()

