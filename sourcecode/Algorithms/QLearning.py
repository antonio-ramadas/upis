#!/usr/bin/python3

from Parser import Parser, DatasetPath
from Headers import ActivityDataHeaders
from sklearn.preprocessing import LabelEncoder
import pandas as pd

"""
Algorithm from:
Reaz, M. B. I., Assim, A., Ibrahimy, M. I., Choong, F., & Mohd-Yasin, F. (2008).
Smart Home Device Usage Prediction Using Pattern Matching and Reinforcement Learning.
7th International Conference on System Identification and Control Problems (SCIPRO’08), (February 2016), 1092–1098.
Retrieved from http://www.econf.info/files/105/1345/1092.pdf
"""


class QLearning:
    def __init__(self, p: Parser, number_of_activities: int):
        self.__p = p
        self.__n_activities = number_of_activities
        self.__encoder = LabelEncoder()

    def algorithm(self):
        # First let's get what we want
        columns = [ActivityDataHeaders.LABEL,
                   ActivityDataHeaders.START_TIME,
                   ActivityDataHeaders.END_TIME]
        activities_df = p.data()[columns]

        label = ActivityDataHeaders.LABEL

        n_most_frequent_activities = activities_df[label].value_counts().index[:self.__n_activities]

        activities_df = activities_df[activities_df[label].isin(n_most_frequent_activities)]

        activities_df.sort_values(ActivityDataHeaders.START_TIME, inplace=True)

        # Build History
        self.__encoder.fit(n_most_frequent_activities)
        initial_state = (False,) * self.__n_activities
        history = {
            initial_state: initial_state
        }




if __name__ == '__main__':
    print('Q-Learning')

    path = DatasetPath.MIT1

    p = Parser(path)

    ql = QLearning(p, 3)

    ql.algorithm()
