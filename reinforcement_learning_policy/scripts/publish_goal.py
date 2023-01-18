#!/usr/bin/env python3

"""
@Author: Miro
@Date: 01/06/2022
@Version: 1.0
@Objective: Publishing visual goal
@TODO:
"""

import rospy
from std_msgs.msg import ColorRGBA
from reinforcement_learning_policy.srv import *
from visualization_msgs.msg import Marker


class GoalRequest:
    def __init__(self):
        self.x_goal = 0
        self.y_goal = 0


goal_req_values = GoalRequest()


def set_goal(req):
    goal_req_values.x_goal, goal_req_values.y_goal = req.goal.x, req.goal.y


def talker():
    pub = rospy.Publisher('goal_position', Marker, queue_size=10)
    rate = rospy.Rate(20)
    while not rospy.is_shutdown():
        pub.publish(marker())
        rate.sleep()


def marker():
    goal_marker = Marker()
    goal_marker.header.frame_id = "map"
    goal_marker.header.seq = 1
    goal_marker.header.stamp = rospy.get_rostime()
    goal_marker.ns = "goal"
    goal_marker.id = 1
    goal_marker.type = goal_marker.SPHERE
    goal_marker.action = goal_marker.ADD
    goal_marker.pose.position.x = goal_req_values.x_goal
    goal_marker.pose.position.y = goal_req_values.y_goal
    goal_marker.color = ColorRGBA(0, 1, 0, 1)
    goal_marker.scale.x = 0.50
    goal_marker.scale.y = 0.50
    return goal_marker


if __name__ == '__main__':
    rospy.init_node('goal_position')
    try:
        rospy.Service('goal_position_to_pub', pub_goal, set_goal)
        talker()
    except rospy.ROSInterruptException:
        exit(1)
