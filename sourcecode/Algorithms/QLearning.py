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
    def __init__(self, parser: Parser):
        self.__parser = parser
        self.__encoder = LabelEncoder()

    def __process(self, dataset: pd.DataFrame) -> pd.DataFrame:
        # First let's get what we want
        columns = [ActivityDataHeaders.LABEL,
                   ActivityDataHeaders.START_TIME,
                   ActivityDataHeaders.END_TIME]
        activities_df = dataset[columns]

        label = ActivityDataHeaders.LABEL

        starts = activities_df[[label, ActivityDataHeaders.START_TIME]]
        ends   = activities_df[[label, ActivityDataHeaders.END_TIME]]
        ends = ends.rename(index=str, columns={ActivityDataHeaders.END_TIME: ActivityDataHeaders.START_TIME})

        activities_df = starts.append(ends)

        # By sorting, when an activity first shows, then it starts, if it shows again, then it is its end
        # We can read now this as a loop
        activities_df.sort_values(ActivityDataHeaders.START_TIME, inplace=True)

        return activities_df

    def __build_graph(self, activities_df: pd.DataFrame) -> dict:
        # Encode the activities so they can act as index
        self.__encoder.fit(activities_df[ActivityDataHeaders.LABEL].unique())

        initial_state      = (False,) * len(self.__encoder.classes_)
        initial_transtions = [False ] * len(self.__encoder.classes_)

        history = {
            initial_state: initial_transtions
        }

        for index, row in activities_df.iterrows():
            activity_idx = self.__encoder.transform([row[ActivityDataHeaders.LABEL]])[0]

            # Update transition
            transitions = history[initial_state]
            transitions[activity_idx] = True
            history[initial_state] = transitions

            # Go to new state
            initial_state = list(initial_state)
            initial_state[activity_idx] = not initial_state[activity_idx]
            initial_state = tuple(initial_state)

            if initial_state not in history:
                history[initial_state] = initial_transtions

        return history

    def algorithm(self):
        activities_df = self.__process(self.__parser.data())

        history = self.__build_graph(activities_df)


if __name__ == '__main__':
    print('Q-Learning')

    # MIT1 has not overlapping activities
    path = DatasetPath.MIT2

    p = Parser(path)

    ql = QLearning(p)

    ql.algorithm()
