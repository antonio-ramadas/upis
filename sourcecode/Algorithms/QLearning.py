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
        self.__history_graph = {}

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

        if self.__history_graph == {}:
            self.__history_graph = {
                initial_state: initial_transtions
            }

        for index, row in activities_df.iterrows():
            activity_idx = self.__encoder.transform([row[ActivityDataHeaders.LABEL]])[0]

            # Update transition
            transitions = self.__history_graph[initial_state]
            transitions[activity_idx] = True
            self.__history_graph[initial_state] = transitions

            # Go to new state
            initial_state = list(initial_state)
            initial_state[activity_idx] = not initial_state[activity_idx]
            initial_state = tuple(initial_state)

            if initial_state not in self.__history_graph:
                self.__history_graph[initial_state] = initial_transtions

    def __pattern_match(self, recent_activities: list) -> int:
        length = 0

        recent_activities = reversed(recent_activities)
        old_state = None

        for new_state in recent_activities:
            if old_state is None:
                if new_state not in self.__history_graph:
                    return length

                old_state = new_state
                continue

            activity = 0
            for i in range(len(new_state)):
                if old_state[i] != new_state[i]:
                    break
                activity += 1

            if new_state in self.__history_graph and self.__history_graph[new_state][activity]:
                length += 1
                old_state = new_state
            else:
                return length

        return length

    def algorithm(self, alpha: float = 0.5):
        assert 0 <= alpha <= 1

        activities_df = self.__process(self.__parser.data())

        self.__build_graph(activities_df)

        number_of_activities = len(activities_df[ActivityDataHeaders.LABEL].unique())

        q = dict.fromkeys(self.__history_graph, [0] * number_of_activities)
        recent_state = (False,) * number_of_activities
        recent_graph = {
            recent_state: [0] * number_of_activities
        }
        recent_activities = [recent_state]

        for index, row in activities_df.iterrows():
            selected_activity = 0
            highest_ranking = -1

            # To prevent division by zero
            activities_sum = max(1, sum(recent_graph[recent_state]))

            for activity in range(number_of_activities):
                new_state = list(recent_state)
                new_state[activity] = not new_state[activity]
                new_state = tuple(new_state)

                l = self.__pattern_match(recent_activities + [new_state])
                r = q[recent_state][activity]

                ranking = (1 - alpha) * l + alpha * (recent_graph[recent_state][activity] / activities_sum + r)

                if ranking > highest_ranking:
                    highest_ranking = ranking
                    selected_activity = activity

            # Update transition
            recent_graph[recent_state][selected_activity] += 1

            # Go to new state
            recent_state = list(recent_state)
            recent_state[selected_activity] = not recent_state[selected_activity]
            recent_state = tuple(recent_state)

            if recent_state not in recent_graph:
                recent_graph[recent_state] = [0] * number_of_activities

            if len(recent_activities) == 0 or recent_state != recent_activities[-1]:
                recent_activities += [recent_state]

            # TODO Update q

            # TODO Not said on the paper, but I could add action to history


if __name__ == '__main__':
    print('Q-Learning')

    # MIT1 has not overlapping activities
    path = DatasetPath.MIT2

    p = Parser(path)

    ql = QLearning(p)

    ql.algorithm()
