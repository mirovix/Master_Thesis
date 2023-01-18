#!/usr/bin/env python3

"""
@Author: Miro
@Date: 06/11/2022
@Version: 1.0
@Objective: Testing file
@TODO:
"""

import sys
import os
import xml.etree.ElementTree as et
from util.DDPG_values import *
from util.ou_action_noise import OUActionNoise

dir_NN_data = "/reinforcement_learning_policy/NN_data/"


def nn_client_test():
    # check iterations in each epoch
    if iter_current_epoch() is True: return 0

    current_state, current_x, current_y = get_processed_input()

    # try an action
    tf_prev_state = tf.expand_dims(tf.convert_to_tensor(norm(current_state)), 0)
    action = policy_definition(tf_prev_state)
    action_x, action_y = action[0][0], action[0][1]

    # set new state
    new_point = setting_sub_goal(current_x, current_y, action)

    # save the path and the min distance for performance measurements
    input_test.avg_path_length[input_test.count_test] += sqrt((action_x ** 2) + (action_y ** 2))
    min_distance = sys.maxint
    for i in range(2, len(current_state), 2):
        obj_relative = np.array([current_state[i], current_state[i + 1]])
        distance = np.linalg.norm(obj_relative)
        if distance < min_distance: min_distance = distance

    if min_distance < input_test.avg_min_distance_obstacles[input_test.count_test]:
        input_test.avg_min_distance_obstacles[input_test.count_test] = min_distance

    rospy.loginfo("Output NN %f,%f", action_x, action_y)
    rospy.loginfo("Position %f,%f", current_x, current_y)
    rospy.loginfo("New Position %f,%f,%f", new_point.x, new_point.y, new_point.theta)

    # send goal and check if it is reached
    try:
        start_time = rospy.get_time()
        sp = rospy.ServiceProxy('robot_goal', NN)
        response = sp(new_point)
        input_test.avg_time[input_test.count_test] += rospy.get_time() - start_time

        new_state, new_x, new_y = get_processed_input()

        save_info_trajectory(input_train.dir_trajectory, new_x, new_y, input_train.epoch, input_train.goal.x,
                             input_train.goal.y, goal_check, response.new_processing)

        if response.new_processing:
            # check if the goal is reached
            if check_goal_reached(new_state) == 0:
                input_train.sequence_goal += 1
                return 1
        else:
            # oscillation due to the reset initial position
            if input_train.t < 2:
                rospy.loginfo("Oscillation detected")
                reset_initial_position_random()
            else:
                return 2
    except rospy.ServiceException as e:
        rospy.loginfo("Service goal call failed: %s" % e)

    return -1


def init_test(directory_weights, directory_trajectory):
    # wait all the services
    wait_services()

    # change max iter and performance
    input_test.erase()

    # update the dir weights
    input_train.update_weights_path(directory_weights)
    input_train.update_trajectory_path(directory_trajectory)

    # reset buffer and noise
    input_train.std_dev = 0.0
    input_train.update_action_noise_buffer(
        OUActionNoise(mean=np.zeros(1), std_deviation=float(input_train.std_dev) * np.ones(1)),
        OUActionNoise(mean=np.zeros(1), std_deviation=float(input_train.std_dev) * np.ones(1)),
        None)

    # reset the lists
    input_train.reset_list()
    input_train.ep_reward_list.append(0.0)
    input_train.ep_critic_list.append(0.0)

    # reset the models
    input_train.update_actor_critic_model()

    # load weights
    load_weights()


def erase_current_information():
    input_test.avg_min_distance_obstacles[input_test.count_test] = 0.0
    input_test.avg_time[input_test.count_test] = 0.0
    input_test.avg_path_length[input_test.count_test] = 0.0


def set_nn_values(path, number_test):
    tree = et.parse(path + "input.xml")
    root = tree.getroot()
    input_train.bool_norm = int(root[number_test - 1][0].text)
    input_train.collision_reward = int(root[number_test - 1][1].text)
    input_train.gamma_reward = int(root[number_test - 1][2].text)

    input_train.bool_norm = 0

    rospy.loginfo("*******")
    rospy.loginfo("normalization %d" % input_train.bool_norm)
    rospy.loginfo("gamma reward %d" % input_train.gamma_reward)
    rospy.loginfo("collision reward %d" % input_train.collision_reward)
    rospy.loginfo("*******\n")


def compute_performance():
    md, tg, pl = 0.0, 0.0, 0.0
    pf = (input_test.performance_acc / (input_test.count_test + 1)) * 100
    for i in range(0, input_test.count_test + 1):
        md += input_test.avg_min_distance_obstacles[i]
        tg += input_test.avg_time[i]
        pl += input_test.avg_path_length[i]

    log_info = "\n*******"
    log_info += "\nTEST NUMBER : %d" % (input_test.count_test + 1)
    log_info += "\nPerformance Accuracy [%%] : %f" % pf
    log_info += "\nNon collision Accuracy [%%] : %f" % (
            (input_test.num_tests - input_test.collision) / input_test.num_tests * 100)
    log_info += "\nAvg path measured [cells] : %f" % (pl / input_test.performance_acc)
    log_info += "\nAvg min distance measured [cells] : %f" % (md / input_test.performance_acc)
    log_info += "\nAvg time [s] : %f" % (tg / input_test.performance_acc)
    log_info += "\n*******\n"

    # write on terminal
    rospy.loginfo(log_info)

    # write on file
    f = open(input_train.dir_weights[:-8] + "result.txt", "w")
    f.write(log_info)
    f.close()


def ddpg_test(directory_models, directory_weights, dir_trajectory, num_model):
    init_test(directory_weights, dir_trajectory)
    rospy.loginfo("*******")
    rospy.loginfo("MODEL NUMBER %d" % num_model)
    rospy.loginfo("*******\n")
    current_status = 0
    input_train.sequence_goal = 0

    # init the test vars
    if directory_models is not None:
        set_nn_values(directory_models, num_model)

    for input_test.count_test in range(0, input_test.num_tests):

        rospy.loginfo("*******")
        rospy.loginfo("TEST NUMBER %d" % input_test.count_test)
        rospy.loginfo("*******\n")

        # reset initial position
        input_train.t = 0
        input_train.epoch = input_test.count_test

        if current_status != 1 or input_train.sequence_goal > input_train.sequence_goal_max:
            input_train.sequence_goal = 0
            reset_initial_position_random()
        goal_inside_occ_grid()

        # try to move toward the goal
        while True:
            current_status = nn_client_test()
            if current_status == 0:
                writeResult("# OF ITER T IS REACHED")
                erase_current_information()
                break
            elif current_status == 1:
                input_test.performance_acc += 1
                writeResult("")
                break
            elif current_status == 2:
                input_test.collision += 1
                writeResult("COLLISION")
                erase_current_information()
                break
        save_data_test(dir_NN_data, str(num_model))
        if input_test.performance_acc > 0:
            compute_performance()


def start():
    rospy.init_node('test_DDPG')
    if len(sys.argv) < 5:
        rospy.loginfo("type of test not defined (all or single) or check the path")
    elif sys.argv[4] == "all":
        dir_models = sys.argv[1]
        os.chdir(dir_models)
        for test in os.listdir():
            if test[:4] == "test":
                num_test = int(test[4:])
                dir_weights = dir_models + test + '/'
                os.chdir(dir_weights)
                for weights in os.listdir():
                    if weights == "weights":
                        dir_weights += weights + '/'
                        ddpg_test(dir_models, dir_weights, sys.argv[1][:-7] + "trajectory/", num_test)
    elif sys.argv[4] == "single":
        ddpg_test(None, sys.argv[1], sys.argv[1][:-7] + "trajectory/", 1)


if __name__ == '__main__':
    start()
