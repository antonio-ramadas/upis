#!/usr/bin/python3

from enum import Enum

class SensorProcessedDataHeaders(Enum):
    def __str__(self):
        return str(self.value)

    ID       = 'SENSOR_ID'
    ACTIVITY = 'ACTIVITY'
    START    = 'START'
    END      = 'END'


class ActivityDataHeaders(Enum):
    def __str__(self):
        return str(self.value)

    LABEL                     = 'ACTIVITY_LABEL'
    DATE                      = 'DATE'
    START_TIME                = 'START_TIME'
    END_TIME                  = 'END_TIME'
    SENSOR_IDS                = 'SENSOR_IDS'
    SENSOR_OBJECTS            = 'SENSOR_OBJECTS'
    SENSOR_ACTIVATION_TIMES   = 'SENSOR_ACTIVATION_TIMES'
    SENSOR_DEACTIVATION_TIMES = 'SENSOR_DEACTIVATION_TIMES'
