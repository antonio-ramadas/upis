#!/usr/bin/python3
from Parser import Parser, DatasetPath
from DataProcessor import DataProcessor
from Headers import ActivityDataHeaders
from Metrics import Metrics
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import numpy as np

"""
Algorithm from:
Reaz, M. B. I., Assim, A., Ibrahimy, M. I., Choong, F., & Mohd-Yasin, F. (2008).
Smart Home Device Usage Prediction Using Pattern Matching and Reinforcement Learning.
7th International Conference on System Identification and Control Problems (SCIPRO’08), (February 2016), 1092–1098.
Retrieved from http://www.econf.info/files/105/1345/1092.pdf
"""


class QLearning:
    def __init__(self, dp: DataProcessor, alpha: float = 0.5, positive_reward: float = 0.5,
                 negative_reward: float = -0.5, discount_factor: float = 0.5):
        assert 0 <= alpha <= 1
        assert 0 <= discount_factor <= 1

        self.__dp = dp
        self.__encoder = LabelEncoder()
        self.__history_graph = {}
        self.__q = {}
        self.__alpha = alpha
        self.__positive_reward = positive_reward
        self.__negative_reward = negative_reward
        self.__discount_factor = discount_factor
        self.__number_of_activities = len(dp.data_processed[ActivityDataHeaders.LABEL].unique())

        # Encode the activities so they can act as index
        self.__encoder.fit(dp.data_processed[ActivityDataHeaders.LABEL].unique())

    def process_dataset(self, activities_df) -> pd.DataFrame:
        """
        Process the dataset into 2 columns: Label and Time it occurred. The dataset as given has 3 columns (Label, Start
        and End). The return is the dataset sorted by time. So the length of an activity is the time between two
        consecutive occurrences. When an activity first shows, then it started. When it shows again, then it ended. Just
        loop between these two states while you traverse the dataset and it is possible to point out the exact state of
        the entire house in any point of time.

        :return: Sorted dataset, by time, with two columns (Label and Time)
        """
        columns = [ActivityDataHeaders.LABEL,
                   ActivityDataHeaders.START_TIME,
                   ActivityDataHeaders.END_TIME]

        label = ActivityDataHeaders.LABEL

        starts = activities_df[[label, ActivityDataHeaders.START_TIME]]
        ends = activities_df[[label, ActivityDataHeaders.END_TIME]]

        ends = ends.rename(index=str, columns={ActivityDataHeaders.END_TIME: ActivityDataHeaders.START_TIME})

        activities_df = starts.append(ends)

        # By sorting, when an activity first shows, then it starts, and when it shows again, then it is its end
        activities_df.sort_values(ActivityDataHeaders.START_TIME, inplace=True)

        return activities_df

    def __build_history_graph(self, activities_df: pd.DataFrame):
        """
        Builds the history graph as detailed on the paper. The graph is written to the object variable

        :param activities_df: Dataset processed
        """
        initial_state = (False,) * self.__number_of_activities
        initial_transitions = [False] * self.__number_of_activities

        self.__history_graph = {
            initial_state: initial_transitions
        }

        for index, row in activities_df.iterrows():
            activity_idx = self.__encoder.transform([row[ActivityDataHeaders.LABEL]])[0]

            # Update transition
            self.__history_graph[initial_state][activity_idx] = True

            # Go to new state
            initial_state = list(initial_state)
            initial_state[activity_idx] = not initial_state[activity_idx]
            initial_state = tuple(initial_state)

            if initial_state not in self.__history_graph:
                self.__history_graph[initial_state] = initial_transitions

    def __pattern_match(self, recent_activities: list) -> int:
        """
        Perform the pattern match as described on the paper

        :param recent_activities: List of the most recent activities (check recent graph description on the paper)
        :return: Length of the pattern match
        """
        length = 0

        # As said on the paper, look backwards
        recent_activities = reversed(recent_activities)
        old_state = None

        for new_state in recent_activities:
            if old_state is None:
                if new_state not in self.__history_graph:
                    return length

                old_state = new_state
                continue

            for activity in range(len(new_state)):
                if old_state[activity] != new_state[activity]:
                    break

            if new_state in self.__history_graph and self.__history_graph[new_state][activity]:
                length += 1
                old_state = new_state
            else:
                return length

        return length

    def __select_activity(self, recent_graph: dict, recent_activities: list, recent_state: tuple):
        """
        Select the activity according to the formula of the paper
        """
        selected_activity = 0
        highest_ranking = -1

        activities_sum = sum(recent_graph[recent_state])

        for activity in range(self.__number_of_activities):
            new_state = list(recent_state)
            new_state[activity] = not new_state[activity]
            new_state = tuple(new_state)

            l = self.__pattern_match(recent_activities + [new_state])
            r = self.__q[recent_state][activity]

            # Following the formula from the paper
            ranking = (1 - self.__alpha) * l
            ranking += self.__alpha * (recent_graph[recent_state][activity] / activities_sum + r)

            if ranking > highest_ranking:
                highest_ranking = ranking
                selected_activity = activity

        return selected_activity

    def __predict(self, activities_df):
        prediction = np.empty((0, 0))

        # Initialize Q matrix
        self.__q = dict.fromkeys(self.__history_graph, [0] * self.__number_of_activities)
        recent_state = (False,) * self.__number_of_activities
        recent_graph = {
            recent_state: [1] * self.__number_of_activities
        }
        recent_activities = [recent_state]

        for index, row in activities_df.iterrows():
            selected_activity = self.__select_activity(recent_graph, recent_activities, recent_state)

            prediction = np.append(prediction, self.__encoder.inverse_transform([selected_activity]))

            # Update transition
            recent_graph[recent_state][selected_activity] += 1

            # Update q before
            is_same_activity = self.__encoder.transform([row[ActivityDataHeaders.LABEL]])[0] == selected_activity
            reward = self.__positive_reward if is_same_activity else self.__negative_reward
            previous_state = recent_state

            # Go to new state
            recent_state = list(recent_state)
            recent_state[selected_activity] = not recent_state[selected_activity]
            recent_state = tuple(recent_state)

            if recent_state not in recent_graph:
                recent_graph[recent_state] = [1] * self.__number_of_activities

            recent_activities += [recent_state]

            self.__q[previous_state][selected_activity] = \
                (1 - self.__alpha) * self.__q[previous_state][selected_activity]
            self.__q[previous_state][selected_activity] += self.__alpha * (
                    reward + self.__discount_factor * max(self.__q[recent_state]))

            # Not said on the paper, but I could add the activities to history

        return prediction

    def fit(self, activities_df: pd.DataFrame):
        """
        Run the algorithm as described on the paper
        """
        activities_df = self.process_dataset(activities_df)

        self.__build_history_graph(activities_df)

        self.__predict(activities_df)

    def evaluate(self, n_folds=10):
        matrices = []
        f1 = 0
        precision = 0
        recall = 0

        for train, test in self.__dp.split(n_folds, ActivityDataHeaders.START_TIME):
            self.fit(train)

            processed_dataset = self.process_dataset(test)

            predictions = self.__predict(processed_dataset)

            metric = Metrics(processed_dataset[ActivityDataHeaders.LABEL], pd.DataFrame(predictions))

            f1 += metric.f1()
            precision += metric.precision()
            recall += metric.recall()
            matrices += [metric.confusion_matrix()]

        f1 /= n_folds
        precision /= n_folds
        recall /= n_folds
        matrices = np.array(matrices)

        return f1, precision, recall, matrices


if __name__ == '__main__':
    print('Q-Learning')

    # MIT1 has not overlapping activities
    path = DatasetPath.MIT2

    dp = DataProcessor(path=path)

    dp.data_processed = Parser().data()

    ql = QLearning(dp)

    ql.fit(dp.data_processed)

    f1, precision, recall, matrices = ql.evaluate()

    print(f'F1        = {f1}')
    print(f'Precision = {precision}')
    print(f'Recall    = {recall}')
