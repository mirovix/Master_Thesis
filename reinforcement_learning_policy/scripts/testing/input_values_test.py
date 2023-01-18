#!/usr/bin/env python3/thesis_mi

"""
@Author: Miro
@Date: 01/06/2022
@Version: 1.0
@Objective: Input values for test the model
@TODO:
"""


class InputValuesTest:
    def __init__(self, num_tests):
        self.num_tests = num_tests
        self.count_test = 0

        self.count_test_policy = 0
        self.current_step = 0

        self.new_input = False

        self.performance_acc = 0
        self.avg_list_temp = [0, 0, 0, 0, 0, 0, 0]
        self.avg_min_list_temp = [float("inf"), float("inf"), float("inf"), float("inf"), float("inf"), float("inf"),
                                  float("inf")]
        self.avg_path_length = []
        self.avg_min_distance_obstacles = []
        for i in range(self.num_tests):
            self.avg_path_length.append(self.avg_list_temp)
            self.avg_min_distance_obstacles.append(self.avg_min_list_temp)

        self.avg_time = [0] * self.num_tests

        self.success = 0
        self.collision = 0

        self.goal_step = [[0.86, -1], [1.63, -4.05], [3.22, -6.41], [1.26, -7.68], [-2.5, -10.7], [1.17, -11.4],
                          [1.09, -7.05]]
        # self.goal_step = [[1.93, -0.85],[1.86, -5.07],[1.74, -7.10],[-1.04, -8.03],[-2.09, -9.50], [-1.25, -11.4],
        # [1.16, -10.03]]

        # self.random_goals_x = np.random.uniform(low=-2.5, high=3, size=(self.num_tests,))
        # self.random_goals_y = np.random.uniform(low=-2.5, high=0.5, size=(self.num_tests,))

    def erase(self):
        self.count_test = 0
        self.performance_acc = 0
        self.avg_path_length = []
        self.avg_min_distance_obstacles = []
        self.avg_time = [0] * self.num_tests
        for i in range(self.num_tests):
            self.avg_path_length.append(self.avg_list_temp)
            self.avg_min_distance_obstacles.append(self.avg_min_list_temp)
        self.success = 0
        self.collision = 0
