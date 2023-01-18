#!/usr/bin/env python3

"""
@Author: Miro
@Date: 10/12/2022
@Version: 1.0
@Objective: user input controller
@TODO:
"""

from geometry_msgs.msg import PointStamped
from test_drone import *
from RL_policy import tf_map_bl_fun


def two_input():
    # 0 left, 1 right
    x, y, x_rl, y_rl = 0, 0, 0, 0
    if sys.argv[2] == "s":
        x = 0.7
        y = 0.5
        x_rl = 0.7
        y_rl = 0.5
    elif sys.argv[2] == "d":
        x = -0.7
        y = -0.5
        x_rl = 0.7
        y_rl = -0.5
    pub = rospy.Publisher('/user_input', PointStamped, queue_size=10)
    pub_rl = rospy.Publisher('/user_input_rl', PointStamped, queue_size=10)
    rate = rospy.Rate(2)
    seq = 0
    point_smp = PointStamped()
    point_smp.header.frame_id = "/map"
    point_smp.point.x = x
    point_smp.point.y = y

    point_smp_tf = tf_map_bl_fun("/base_link", 0, Pose2D(x_rl, y_rl, 0))
    point_smp_rl = PointStamped()
    point_smp_rl.header.frame_id = "/map"
    point_smp_rl.point.x = point_smp_tf.point.x
    point_smp_rl.point.y = point_smp_tf.point.y
    point_smp_rl.point.z = -10

    for i in range(0, 2):
        pub_rl.publish(point_smp_rl)
        pub.publish(point_smp)
        seq += 1
        rate.sleep()
    point_smp.point.x = 0
    point_smp.point.y = 0
    sleep(1.5)
    for i in range(0, 2):
        pub.publish(point_smp)
        seq += 1
        rate.sleep()
    rospy.loginfo("publish input command")


def single_input():
    # 0 left, 1 right
    x_rl = 1.7
    y_rl = 0.0
    pub_rl = rospy.Publisher('/user_input_rl', PointStamped, queue_size=10)
    rate = rospy.Rate(2)
    seq = 0

    point_smp_tf = tf_map_bl_fun("/base_link", 0, Pose2D(x_rl, y_rl, 0))
    point_smp_rl = PointStamped()
    point_smp_rl.header.frame_id = "/map"
    point_smp_rl.point.x = point_smp_tf.point.x
    point_smp_rl.point.y = point_smp_tf.point.y

    for i in range(0, 2):
        pub_rl.publish(point_smp_rl)
        seq += 1
        rate.sleep()


if __name__ == '__main__':
    rospy.init_node('input_by_cmd')
    if float(sys.argv[1]) == 1:
        single_input()
    else:
        two_input()
