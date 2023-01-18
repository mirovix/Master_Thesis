#!/usr/bin/env python3

"""
@Author: Miro
@Date: 20/05/2022
@Version: 1.1
@Objective: Cluster analysis of the map through different algorithm
@TODO:
"""

import rospy
from sklearn.cluster import MeanShift, DBSCAN, Birch, estimate_bandwidth
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from nav_msgs.msg import OccupancyGrid

dir_name = sys.argv[1]
os.chdir(os.path.dirname(dir_name))


class InputData:
    def __init__(self, info, current_seq):
        self.info = info
        self.current_seq = current_seq

    def update(self, info, current_seq):
        self.info = info
        self.current_seq = current_seq


map_info = InputData(OccupancyGrid, -1)


def run(seq, threshold=95):
    size_x = map_info.info.info.width
    size_y = map_info.info.info.height
    input_cluster = []
    for i in range(size_y):
        for j in range(size_x):
            if map_info.info.data[i + j * size_x] > threshold:
                input_cluster.append([i, j])
    if len(input_cluster) == 0: return
    input_cluster = np.asarray(input_cluster)

    start = rospy.get_time()
    bir = Birch(threshold=25, n_clusters=None).fit(input_cluster).predict(input_cluster)
    end = rospy.get_time()
    rospy.loginfo("Time for computing BIRCH ", (end - start))

    start = rospy.get_time()
    mean_shift = MeanShift(bandwidth=estimate_bandwidth(input_cluster, quantile=0.2), bin_seeding=True)\
        .fit(input_cluster)
    end = rospy.get_time()
    rospy.loginfo("Time for computing MEAN SHIFT ", (end - start))

    start = rospy.get_time()
    DBSCAN(eps=2, min_samples=10).fit(input_cluster)
    end = rospy.get_time()
    rospy.loginfo("Time for computing DBSCAN ", (end - start))

    start = rospy.get_time()
    cp_list = closest_point(bir, input_cluster, size_x / 2, size_y / 2)
    end = rospy.get_time()
    rospy.loginfo("Time for computing CLOSEST POINT ", (end - start))

    plot_map(np.asarray(map_info.info.data).reshape(size_x, size_y), seq)
    plot_birch(input_cluster, bir, seq, np.asarray(cp_list))
    plot(mean_shift, input_cluster, mean_shift.labels_, seq)


def closest_point(labels_, x, robot_x, robot_y):
    n_clusters_ = len(np.unique(labels_))
    rob_pos = np.array((robot_x, robot_y))
    cp_list = []
    for i in range(n_clusters_):
        min_x, min_y, min_dist = sys.maxsize, sys.maxsize, sys.maxsize
        for p in x[np.where(labels_ == i)]:
            dist = np.linalg.norm(p - rob_pos)
            if dist < min_dist:
                min_x, min_y, min_dist = p[0], p[1], dist
        cp_list.append([min_x, min_y])
    return cp_list


def plot(cluster, x, label, j):
    plt.clf()
    for i in range(len(np.unique(label))):
        plt.scatter(x[label == i, 0], x[label == i, 1], s=1, label=i)
    plt.scatter(cluster.cluster_centers_[:, 0], cluster.cluster_centers_[:, 1], s=80, color='k')
    plt.scatter([80], [80], s=80, color='k')
    plt.xlim([0, 161])
    plt.ylim([0, 161])

    name = "ms_" + str(j) + ".png"
    plt.savefig(name)


def plot_birch(x, labels_, j, cp_list):
    plt.clf()
    for i in range(len(np.unique(labels_))):
        plt.scatter(x[labels_ == i, 0], x[labels_ == i, 1], s=1, label=i)
    plt.scatter(cp_list[:, 0], cp_list[:, 1], s=80, label=0)
    plt.scatter([80], [80], s=80, color='k')
    plt.xlim([0, 161])
    plt.ylim([0, 161])

    name = "birch_" + str(j) + ".png"
    plt.savefig(name)


def plot_map(data, j):
    plt.clf()
    plt.pcolor(data, linewidths=1)
    plt.xlim([0, 161])
    plt.ylim([0, 161])
    name = "map_" + str(j) + ".png"
    plt.savefig(name)


def callback(data):
    rospy.loginfo("New cost map detected, sequence ", data.header.seq)
    map_info.update(data, data.header.seq)
    run(data.header.seq)


def cluster_map():
    rospy.init_node('cluster_map', anonymous=True)
    rospy.Subscriber("/move_base/local_costmap/costmap", OccupancyGrid, callback)
    rospy.spin()


if __name__ == '__main__':
    cluster_map()
