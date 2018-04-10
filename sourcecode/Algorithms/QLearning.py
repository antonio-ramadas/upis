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
        self.n_most_frequent_activities = []

    def __process(self, dataset: pd.DataFrame) -> pd.DataFrame:
        # First let's get what we want
        columns = [ActivityDataHeaders.LABEL,
                   ActivityDataHeaders.START_TIME,
                   ActivityDataHeaders.END_TIME]
        activities_df = dataset[columns]

        label = ActivityDataHeaders.LABEL

        # Get the most frequent activities
        self.n_most_frequent_activities = activities_df[label].value_counts().index[:self.__n_activities]

        activities_df = activities_df[activities_df[label].isin(self.n_most_frequent_activities)]

        starts = activities_df[[label, ActivityDataHeaders.START_TIME]]
        ends   = activities_df[[label, ActivityDataHeaders.END_TIME]]
        ends.rename(index=str, columns={ActivityDataHeaders.END_TIME: ActivityDataHeaders.START_TIME}, inplace=True)

        activities_df = starts.append(ends)

        # By sorting, when an activity first shows, then it starts, if it shows again, then it is its end
        # We can read now this as a loop
        activities_df.sort_values(ActivityDataHeaders.START_TIME, inplace=True)

        return activities_df

    def algorithm(self):
        activities_df = self.__process(p.data())

        # Build History
        self.__encoder.fit(self.n_most_frequent_activities)
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
