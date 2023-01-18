#!/usr/bin/env python3

import math
import pickle
import random
from math import sqrt
from time import sleep
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import rospy
from reinforcement_learning_policy.srv import *
from gazebo_msgs.msg import ModelState
from gazebo_msgs.srv import SetModelState
from geometry_msgs.msg import Pose2D, PoseWithCovarianceStamped
from nav_msgs.msg import OccupancyGrid
from nav_msgs.srv import SetMap

import sys
sys.path.insert(1, '/reinforcement_learning_policy/scripts/')

from training import config_train as ct
from testing.input_values_test import InputValuesTest
from training.input_values_train import InputValuesTrain

input_train = InputValuesTrain()

input_test = InputValuesTest(3)


def reward_function(input_reward, alfa=input_train.alfa_reward, gamma=input_train.gamma_reward, k=input_train.k,
                    l_max=input_train.l_max,
                    d_infl=input_train.d_infl):
    # x and y relative position of the aerial robot with respect to the desired goal
    p_goal = np.linalg.norm(np.array([input_reward[0], input_reward[1]]))
    u_att = alfa * p_goal
    if p_goal > d_infl:
        beta = gamma
    else:
        beta = gamma / (np.exp(4 * (d_infl - p_goal)))

    obstacles = 0
    for i in range(2, len(input_reward), 2):
        if input_reward[i] >= l_max and input_reward[i + 1] >= l_max:
            continue
        obj_relative = np.array([input_reward[i], input_reward[i + 1]])
        distance = np.linalg.norm(obj_relative)
        obstacles += (1 / (k + distance)) - (1 / (k + l_max))

    u_rep = beta * obstacles

    shaping_current = - u_att - u_rep
    return shaping_current


# this update target parameters slowly
# based on rate `tau`, which is much less than one.
@tf.function
def update_target(target_weights, weights, tau):
    for (a, b) in zip(target_weights, weights):
        a.assign(b * tau + a * (1 - tau))


def policy_definition(state):
    sampled_actions = tf.squeeze(input_train.actor_model(state))
    noise = [input_train.ou_noise_x().tolist()[0], input_train.ou_noise_y().tolist()[0]]
    rospy.loginfo(noise)

    sampled_actions = sampled_actions.numpy()
    sampled_actions[0] += noise[0]
    sampled_actions[1] += noise[1]

    # we make sure action is within bounds
    # clipping min values
    for i in range(len(sampled_actions)):
        if 0.0 < sampled_actions[i] < ct.min_clip_value:
            sampled_actions[i] = ct.min_clip_value
        elif 0.0 > sampled_actions[i] > -ct.min_clip_value:
            sampled_actions[i] = -ct.min_clip_value

    legal_action = np.clip(sampled_actions, input_train.lower_bound, input_train.upper_bound)

    return [np.squeeze(legal_action)]


def goal_inside_occ_grid():
    resp_goal_server = None
    try:
        goal_server = rospy.ServiceProxy('goal_setting', goal)
        resp_goal_server = goal_server(True)
    except rospy.ServiceException as e:
        rospy.loginfo("Service pre processing call failed: %s" % e)
        exit(1)

    # save the data received
    input_train.goal.x = resp_goal_server.goal.data[0]
    input_train.goal.y = resp_goal_server.goal.data[1]

    rospy.loginfo("Random goal %f,%f", input_train.goal.x, input_train.goal.y)
    publish_goal()


def goal_setting():
    # in_NN.goal.y = random.uniform(-2.5, 0.5)
    # in_NN.goal.x = random.uniform(-3.5, 3.5)
    # random goal on the map
    # if randrange(10)>3:
    input_train.goal.x = random.uniform(7.2, 13)
    input_train.goal.y = random.uniform(-3.8, 1.2)
    # else:
    #    in_NN.goal.x = random.uniform(0,6)
    #    in_NN.goal.y = random.uniform(-0.5,0.5)
    rospy.loginfo("RANDOM GOAL %f,%f", input_train.goal.x, input_train.goal.y)
    publish_goal()


def publish_goal():
    try:
        sp = rospy.ServiceProxy('goal_position_to_pub', pub_goal)
        sp(Pose2D(input_train.goal.x, input_train.goal.y, 0))
    except rospy.ServiceException as e:
        rospy.loginfo("Service goal call failed: %s" % e)


def check_goal_reached(state):
    if sqrt(pow(state[0], 2) + pow(state[1], 2)) < ct.goal_check:
        rospy.loginfo("*******")
        rospy.loginfo("*******")
        rospy.loginfo("*******")
        rospy.loginfo("*******")
        rospy.loginfo("GOAL REACHED")
        rospy.loginfo("*******")
        rospy.loginfo("*******")
        rospy.loginfo("*******")
        rospy.loginfo("*******\n")
        return 0
    return 1


def setting_sub_goal(current_x, current_y, action):
    # set the middle goal

    point = Pose2D()
    point.x = action[0][0] + current_x
    point.y = action[0][1] + current_y
    point.theta = np.arctan2((point.y - current_y), (point.x - current_x))

    return point


def wait_services():
    rospy.wait_for_service('/robot_goal')
    rospy.wait_for_service('/robot_data')
    rospy.wait_for_service('/goal_setting')
    rospy.wait_for_service('/set_map')
    rospy.wait_for_service('/gazebo/set_model_state')


def iter_current_epoch():
    input_train.t += 1

    # check iterations in each epoch
    if input_train.t > input_train.max_t: return True

    reduce_error_of_positioning()

    rospy.loginfo("*******")
    rospy.loginfo("Iteration %d" % input_train.t)
    rospy.loginfo("*******\n")
    return False


def compute_q_value(prev_state, action):
    state = tf.expand_dims(tf.convert_to_tensor(norm(prev_state)), 0)
    action_exp = tf.expand_dims(tf.convert_to_tensor(norm(action)), 0)
    return tf.squeeze(input_train.critic_model([state, action_exp])).numpy()


def print_compute_shaping(state, prev_state, exp_reward):
    rospy.loginfo("******")
    for value in state: rospy.loginfo("State %f" % value)
    rospy.loginfo("******\n")

    if exp_reward == input_train.collision_reward or exp_reward == input_train.goal_reward:
        return exp_reward
    else:
        current_shaping = reward_function(state)
        previous_shaping = reward_function(prev_state)
        return current_shaping - previous_shaping


def reset_initial_position_random():
    value, current_orientation = random.randint(0, 1), None
    if value == 0:
        pos_x = random.uniform(7.5, 8.8)
        pos_y = random.uniform(-2, -1.12)
    elif value == 1:
        pos_x = random.uniform(10.5, 12.5)
        pos_y = random.uniform(-2, -1.12)

    pos_x, pos_y = -4, -1

    try:
        current_orientation = (rospy.wait_for_message('/robot_pose', PoseWithCovarianceStamped)).pose.pose.orientation
    except rospy.ServiceException as e:
        rospy.loginfo("Service goal to starting position call failed: %s" % e)

    try:
        sp = rospy.ServiceProxy('robot_goal', NN)
        t3 = +2.0 * (current_orientation.w * current_orientation.z + current_orientation.x * current_orientation.y)
        t4 = +1.0 - 2.0 * (current_orientation.y ** 2 + current_orientation.z ** 2)
        yaw_z = math.atan2(t3, t4)
        sp(Pose2D(pos_x, pos_y, yaw_z))
    except rospy.ServiceException as e:
        rospy.loginfo("Service goal to starting position call failed: %s" % e)

    state_msg = ModelState()
    state_msg.model_name = 'tiago'
    state_msg.pose.position.x = pos_x  # - 6.82
    state_msg.pose.position.y = pos_y  # + 1.37
    state_msg.pose.orientation = current_orientation

    initial = PoseWithCovarianceStamped()
    initial.header.frame_id = 'map'
    initial.pose.pose.position.x = pos_x
    initial.pose.pose.position.y = pos_y
    initial.pose.pose.orientation = current_orientation

    try:
        set_state = rospy.ServiceProxy('/gazebo/set_model_state', SetModelState)
        set_state(state_msg)
    except rospy.ServiceException as e:
        rospy.loginfo("Service goal to starting position call failed: %s" % e)

    try:
        sleep(1)
        set_map = rospy.ServiceProxy('/set_map', SetMap)
        input_train.init_occ = rospy.wait_for_message('/move_base/global_costmap/costmap', OccupancyGrid)
        set_map(input_train.init_occ, initial)
    except rospy.ServiceException as e:
        rospy.loginfo("Service goal to starting position call failed: %s" % e)

    sleep(1.5)
    return


def reduce_error_of_positioning():
    try:
        position = rospy.wait_for_message('/robot_pose', PoseWithCovarianceStamped)
        set_map = rospy.ServiceProxy('/set_map', SetMap)
        set_map(rospy.wait_for_message('/move_base/local_costmap/costmap', OccupancyGrid), position)

        state_msg = ModelState()
        state_msg.model_name = 'tiago'
        state_msg.pose = position.pose.pose
        state_msg.pose.position.x = position.pose.pose.position.x  # - 6.82
        state_msg.pose.position.y = position.pose.pose.position.y  # + 1.37

        set_state = rospy.ServiceProxy('/gazebo/set_model_state', SetModelState)
        set_state(state_msg)
    except rospy.ServiceException as e:
        rospy.loginfo("Service goal to starting position call failed: %s" % e)


def get_processed_input():
    input_list, resp_pre_processing = [], None
    try:
        pre_processing_server = rospy.ServiceProxy('robot_data', preprocessing)
        resp_pre_processing = pre_processing_server(True)
    except rospy.ServiceException as e:
        rospy.loginfo("Service pre processing call failed: %s" % e)

    # save the data received
    for i in range(0, input_train.num_states):
        input_list.append(resp_pre_processing.input_vector.data[i])

    # current position of the robot
    current_x, current_y = input_list[0], input_list[1]

    input_list[0] -= input_train.goal.x
    input_list[1] -= input_train.goal.y

    return input_list, current_x, current_y


def save_data_training(path):
    filehandler = open(path + "in_NN.obj", "wb")
    file_to_save = [input_train.collision_count, input_train.win_count, input_train.ep_critic_list,
                    input_train.ep_reward_list, input_train.avg_reward_list, input_train.avg_reward_list_200,
                    input_train.critic_list, input_train.critic_list_200, input_train.epoch + 1]
    pickle.dump(file_to_save, filehandler)
    filehandler.close()
    rospy.loginfo("in_NN data saved\n")


def save_data_test(path, model_name):
    filehandler = open(path + "in_NN_test_" + model_name + ".obj", "wb")
    file_to_save = [input_test.collision, input_test.success, input_test.avg_time, input_test.avg_path_length,
                    input_test.avg_min_distance_obstacles, input_test.performance_acc, input_test.num_tests]
    pickle.dump(file_to_save, filehandler)
    filehandler.close()
    rospy.loginfo("in_NN_test data saved\n")


def load_data(path):
    filehandler = open(path + "in_NN.obj", "rb")
    input_train.collision_count, input_train.win_count, input_train.ep_critic_list, \
    input_train.ep_reward_list, input_train.avg_reward_list, input_train.avg_reward_list_200, \
    input_train.critic_list, input_train.critic_list_200, \
    input_train.epoch = pickle.load(filehandler)
    filehandler.close()
    rospy.loginfo("in_NN loaded\n")


def norm(input_not_norm):
    return input_not_norm


def plot_reward_value(avg_reward_list, name):
    plt.figure(figsize=(10, 6))
    plt.plot(avg_reward_list)
    plt.title(name)
    plt.savefig(name)


def plot_q_value(avg_q_list, name):
    plt.figure(figsize=(10, 6))
    plt.plot(avg_q_list)
    plt.title(name)
    plt.savefig(name)


def save_weights():
    input_train.actor_model.save_weights(input_train.dir_weights + "drone_actor.h5")
    input_train.critic_model.save_weights(input_train.dir_weights + "drone_critic.h5")
    input_train.target_actor.save_weights(input_train.dir_weights + "drone_target_actor.h5")
    input_train.target_critic.save_weights(input_train.dir_weights + "drone_target_critic.h5")


def load_weights():
    try:
        input_train.actor_model.load_weights(input_train.dir_weights + "drone_actor.h5")
        input_train.critic_model.load_weights(input_train.dir_weights + "drone_critic.h5")
        input_train.target_actor.load_weights(input_train.dir_weights + "drone_target_actor.h5")
        input_train.target_critic.load_weights(input_train.dir_weights + "drone_target_critic.h5")
        rospy.loginfo("Weights load successfully")
    except Exception as ex:
        rospy.loginfo("Cannot find the weights ", ex)


def save_info_trajectory(path, x, y, epoch, x_goal, y_goal, goal_check_value, state):
    if input_train.t == 1 and input_train.epoch == 0:
        char = "w"
    else:
        char = "a"

    if state:
        num_state = 1
    else:
        num_state = 0

    f = open(path + "info_test_policy.txt", char)
    f.write(str(x) + " " + str(y) + " " + str(epoch) + " " +
            str(x_goal) + " " + str(y_goal) + " " + str(goal_check_value) + " " +
            str(num_state) + "\n")
    f.close()
