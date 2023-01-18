#!/usr/bin/env python3

"""
@Author: Miro
@Date: 01/06/2022
@Version: 1.1
@Objective: Dynamic obstacles for testing phase
@TODO:
"""

import sys
import rospy
import numpy as np
from time import sleep
from gazebo_msgs.msg import ModelState
from gazebo_msgs.srv import SetModelState


# dynamic obstacles for model testing (environment 1)
def env_1():
    while True:
        for i in np.arange(0.0, 3.0, 0.1):
            state_msg = model_definition('cabinet_1', -1.5, i - 2)

            state_msg_2 = model_definition('cabinet_2', i, 1)

            state_msg_3 = model_definition('cabinet_3', 2.5, -i + 1)

            state_msg_4 = model_definition('cabinet_4', i / 4, i / 4 - 2)

            publish_models([state_msg, state_msg_2, state_msg_3, state_msg_4])

            sleep(3.5)


# dynamic obstacles for model testing (environment 2)
def env_2():
    while True:
        for i in np.arange(0.0, 3.0, 0.1):
            state_msg = model_definition('cabinet_1', i - 1, -1.7)

            state_msg_2 = model_definition('cabinet_2', i - 0.4, -1.7)

            state_msg_3 = model_definition('cabinet_3', 3, -i + 1)

            state_msg_4 = model_definition('cabinet_4', 3, -i + 0.4)

            publish_models([state_msg, state_msg_2, state_msg_3, state_msg_4])

            sleep(3.5)


# dynamic obstacles for final path testing (combing shared intelligence with RL model)
def env_test_policy():
    while True:
        for i in np.arange(0.0, 3.0, 0.1):
            state_msg = model_definition('cabinet', -1.767812 + i, -1.796058)

            state_msg_2 = model_definition('cabinet_2', 4.312303, -6.395211 - i)

            state_msg_3 = model_definition('cabinet_1', -1.825406 + i, -6.983367)

            state_msg_4 = model_definition('cabinet_0', -4.282325 + i / 3, -11.75668 + i / 3)

            publish_models([state_msg, state_msg_2, state_msg_3, state_msg_4])

            sleep(2.5)


def model_definition(name, x, y):
    state_msg = ModelState()
    state_msg.model_name = name
    state_msg.pose.position.x = x
    state_msg.pose.position.y = y
    return state_msg


def publish_models(list_model):
    for e in list_model:
        try:
            set_state = rospy.ServiceProxy('/gazebo/set_model_state', SetModelState)
            set_state(e)
        except rospy.ServiceException as e:
            rospy.loginfo("Service goal to starting position call failed: %s" % e)


if __name__ == '__main__':
    if sys.argv[1] == '1': env_1()
    elif sys.argv[1] == '2': env_2()
    elif sys.argv[1] == '3': env_test_policy()
    else: exit(1)
