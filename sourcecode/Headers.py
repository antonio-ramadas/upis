#!/usr/bin/python3

from enum import Enum, auto


class SensorProcessedDataHeaders(Enum):
    """
    This enumerable contains the headers (columns) of the processed sensors.csv file
    """
    def __str__(self):
        """
        :return: The string of the Enum's value
        """
        return str(self.value)

    ID       = 'SENSOR_ID'
    ACTIVITY = 'ACTIVITY'
    START    = 'START'
    END      = 'END'


class ActivityDataHeaders(Enum):
    """
    This enumerable contains the headers (columns) of the original activities_data.csv file
    """
    def __str__(self):
        """
        :return: The string of the Enum's value
        """
        return str(self.value)

    LABEL                     = 'ACTIVITY_LABEL'
    DATE                      = 'DATE'
    START_TIME                = 'START_TIME'
    END_TIME                  = 'END_TIME'
    SENSOR_IDS                = 'SENSOR_IDS'
    SENSOR_OBJECTS            = 'SENSOR_OBJECTS'
    SENSOR_ACTIVATION_TIMES   = 'SENSOR_ACTIVATION_TIMES'
    SENSOR_DEACTIVATION_TIMES = 'SENSOR_DEACTIVATION_TIMES'


class NaiveBayesType(Enum):
    """
    This enumerable lists the existing implemented types of Naive Bayes
    """
    SINGLE   = auto()
    MULTIPLE = auto()
