#!/usr/bin/env python3

"""
@Author: Miro
@Date: 01/06/2022
@Version: 1.0
@Objective: Preprocessing occupancy grid into states
@TODO:
"""

import os
import random
from random import randint
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import Birch
import rospy
from reinforcement_learning_policy.srv import *
from geometry_msgs.msg import Twist, Point, PoseWithCovarianceStamped
from nav_msgs.msg import OccupancyGrid
from std_msgs.msg import Float64MultiArray

dir_name = sys.argv[1] + "/thesis_mihailovic/reinforcement_learning_policy/plots/"
os.chdir(os.path.dirname(dir_name))
plt.clf()


class PreProcessingData:
    def __init__(self, velocity, position, nav_msg, prev_sequence):
        self.velocity = velocity
        self.position = position
        self.nav_msg = nav_msg
        self.prev_sequence = prev_sequence
        self.goal = [0, 0]


pre_proc_input_data = PreProcessingData(Twist(), Point(), OccupancyGrid(), 1)


def plot_clusters(x, labels_, cp_list, name="birch.png"):
    plt.clf()
    n_clusters_ = len(np.unique(labels_))
    if n_clusters_ == 0:
        return
    for i in range(0, n_clusters_):
        plt.scatter(x[labels_ == i, 0], x[labels_ == i, 1], s=1, label=i)
    plt.scatter(cp_list[:, 0], cp_list[:, 1], s=80, label=0)
    plt.scatter([80], [80], s=80, color='k')
    plt.xlim([0, 161])
    plt.ylim([0, 161])
    plt.savefig(name)


def map_callback(data):
    pre_proc_input_data.nav_msg = data
    rospy.loginfo("new cost map detected, sequence:%d", data.header.seq)


def call_robot_pose():
    robot_pose = rospy.wait_for_message('/robot_pose', PoseWithCovarianceStamped)
    return robot_pose.pose.pose.position.x, robot_pose.pose.pose.position.y


# function used for determine the closest point between a cluster (i.e. set of object) and the robot
def compute_closest_points(labels_, data):
    dist_inf_radius = 0  # (inf_radius*0.8)/2

    n_clusters_ = len(np.unique(labels_))
    size_x = pre_proc_input_data.nav_msg.info.width
    size_y = pre_proc_input_data.nav_msg.info.height
    new_scale = 0.8 / (size_x / 2)
    rob_pos = np.array((size_x / 2, size_y / 2))

    cp_list_dist, cp_list, cp_plot = [], [], []
    for i in range(0, n_clusters_):
        min_x, min_y, min_dist = sys.maxsize, sys.maxsize, sys.maxsize
        for p in data[np.where(labels_ == i)]:
            dist = np.linalg.norm(p - rob_pos)
            if dist < min_dist:
                min_x = p[0]
                min_y = p[1]
                min_dist = dist
        new_distance = (min_dist * new_scale) - dist_inf_radius
        if new_distance > 0:
            cp_list_dist.append((min_dist * new_scale) - dist_inf_radius)
        else:
            cp_list_dist.append(0.0)

        cp_plot.append([min_x, min_y])
        cp_list.append(((min_x - rob_pos[0]) * new_scale) - dist_inf_radius)
        cp_list.append(((min_y - rob_pos[1]) * new_scale) - dist_inf_radius)

    # manage empty cells
    for i in range(n_clusters_, 6):
        cp_list_dist.append((size_x / 2) * new_scale)
        cp_list.append((size_x / 2) * new_scale)
        cp_list.append((size_x / 2) * new_scale)

    return cp_plot, cp_list, cp_list_dist


def clusterize_map(threshold_map=10, threshold_birch=15):
    size_x = pre_proc_input_data.nav_msg.info.width
    size_y = pre_proc_input_data.nav_msg.info.height
    input_cluster = []

    for i in range(0, size_y):
        for j in range(0, size_x):
            if pre_proc_input_data.nav_msg.data[i + j * size_x] > threshold_map:
                input_cluster.append([i, j])
    if len(input_cluster) == 0:
        return [np.linalg.norm((size_x / 2) * (0.8 / (size_x / 2)))] * 12
    input_cluster = np.asarray(input_cluster)

    start = rospy.get_time()
    bir = Birch(threshold=threshold_birch, n_clusters=None).fit(input_cluster).predict(input_cluster)
    rospy.loginfo("time for computing BIRCH %f", (rospy.get_time() - start))

    start = rospy.get_time()
    cp_plot, cp_list, cp_list_dist = compute_closest_points(bir, input_cluster)
    rospy.loginfo("time for computing CLOSEST POINT %f", (rospy.get_time() - start))

    # plotBirch(input_cluster, bir, np.asarray(cp_plot))

    return cp_list


def goal_setting():
    output = Float64MultiArray()

    # take random goal that is reachable
    size_x = pre_proc_input_data.nav_msg.info.width
    size_y = pre_proc_input_data.nav_msg.info.height
    new_scale = 0.8 / (size_x / 2)

    while True:
        pre_proc_input_data.goal = [randint(0, size_x - 1), randint(0, size_y - 1)]
        if pre_proc_input_data.nav_msg.data[(pre_proc_input_data.goal[1]) +
                                            (pre_proc_input_data.goal[0]) * size_x] == 0:
            break

    # transform goal into global coords
    current_pos = call_robot_pose()
    output.data = [current_pos[0] + ((random.choice([-1, 1])) * (pre_proc_input_data.goal[0] * new_scale)),
                   current_pos[1] + ((random.choice([-1, 1])) * (pre_proc_input_data.goal[1] * new_scale))]
    rospy.loginfo("goal x %f goal y %f", output.data[0], output.data[1])

    return goalResponse(output)


def pre_processing():
    # def output of the array
    input_pose = [0, 0]
    output = Float64MultiArray()
    pre_proc_input_data.prev_sequence -= 1

    input_pose[0], input_pose[1] = call_robot_pose()
    output.data = input_pose + clusterize_map()

    for c in output.data: rospy.loginfo("Value: %f", c)
    rospy.loginfo("Information goal is sent")

    return preprocessingResponse(output)


def pre_processing_server():
    rospy.init_node('pre_processing', anonymous=True)
    rospy.Subscriber("/move_base/local_costmap/costmap", OccupancyGrid, map_callback)

    rospy.Service('robot_data', preprocessing, pre_processing)
    rospy.Service('goal_setting', goal, goal_setting)
    rospy.spin()


if __name__ == '__main__':
    pre_processing_server()
