#!/usr/bin/env python3

"""
@Author: Miro
@Date: 06/11/2022
@Version: 1.0
@Objective: Testing file for RL policy
@TODO:
"""

import xml.etree.ElementTree as ET
from geometry_msgs.msg import PointStamped, PoseStamped
import os
from util.DDPG_values import *
from util.ou_action_noise import OUActionNoise
from policies_system.srv import GoalProvider as GoalProviderSrv
from test_drone import ddpg_test, start
import sys

sys.path.insert(1, '/')
dir_NN_data = "//"


def callback_input(data):
    rospy.loginfo("info message")
    if data.point.z == -10:
        input_train.count_input += 1
    input_train.goal.x = data.point.x
    input_train.goal.y = data.point.y
    input_test.new_input = True


def callback_goal(msg_goal):
    input_train.goal_policy = Pose2D(msg_goal.pose.position.x, msg_goal.pose.position.y, 0)
    input_train.exit_policy = 1


def setting_sub_goal_policy(new_point):
    ps_rl_input = PointStamped()
    ps_rl_input.header.frame_id = "/map"
    ps_rl_input.point.x = new_point.x
    ps_rl_input.point.y = new_point.y
    input_train.RL_pub.publish(ps_rl_input)

    try:
        app_pol = rospy.ServiceProxy('/apply_policies', GoalProviderSrv)
        app_pol()
    except rospy.ServiceException as e:
        rospy.loginfo("Service goal call failed: %s" % e)

    while True:
        if input_train.exit_policy == 1:
            break
    input_train.exit_policy = 0
    return input_train.goal_policy


def goal_policy_check():
    _, x, y = get_processed_input()
    for i in range(input_test.current_step, len(input_test.goal_step)):
        a = np.array([input_test.goal_step[i][0], input_test.goal_step[i][1]])
        b = np.array([x, y])
        if np.linalg.norm(a - b) < goal_check * 1.5:
            rospy.loginfo("\n\n\n\n***\n STEP GOAL REACHED \n\n\n\n\n***")
            input_train.t = 0
            input_test.performance_acc += 1
            input_test.current_step += 1
        break
    return 0


def move_to_prev_goal():
    if input_test.current_step == 6: return
    back_pose = Pose2D(input_test.goal_step[input_test.current_step][0],
                       input_test.goal_step[input_test.current_step][1], 0)
    _, x, y = get_processed_input()
    back_pose.theta = np.arctan2((back_pose.y - y), (back_pose.x - x))
    try:
        sp = rospy.ServiceProxy('robot_goal', NN)
        sp(back_pose)
    except rospy.ServiceException as e:
        rospy.loginfo("Service goal call failed: %s" % e)


if __name__ == '__main__':
    start()
