#!/usr/bin/env python3

"""
@Author: Miro
@Date: 20/05/2022
@Version: 1.0
@Objective: Training file
@TODO:
"""

import sys

sys.path.insert(1, '/reinforcement_learning_policy/scripts/util')

from buffer import Buffer
from ou_action_noise import OUActionNoise
from DDPG_values import *

dir_weights = sys.argv[1] + "/thesis_mihailovic/reinforcement_learning_policy/weights/"
dir_plots = sys.argv[1] + "/thesis_mihailovic/reinforcement_learning_policy/plots/"
dir_trajectory = sys.argv[1] + "/thesis_mihailovic/reinforcement_learning_policy/trajectory/"
dir_NN_data = sys.argv[1] + "/thesis_mihailovic/reinforcement_learning_policy/NN_data/"

# update the dir plots
input_train.update_plot_path(dir_plots)
input_train.update_weights_path(dir_weights)
input_train.update_trajectory_path(dir_trajectory)

# init the buffer and action noise
input_train.update_action_noise_buffer(
    OUActionNoise(mean=np.zeros(1), std_deviation=float(input_train.std_dev) * np.ones(1)),
    OUActionNoise(mean=np.zeros(1), std_deviation=float(input_train.std_dev) * np.ones(1)),
    Buffer(input_train.buffer_size, input_train.batch_size))

# create the network
input_train.update_actor_critic_model()


def train(state, action_x, action_y, prev_state, exp_reward):
    reward = print_compute_shaping(state, prev_state, exp_reward)
    q_value = compute_q_value(prev_state, [action_x, action_y])

    # save the information
    input_train.buffer.record((norm(prev_state), norm([action_x, action_y]), reward, norm(state)))

    # update
    input_train.buffer.learn()
    update_target(input_train.target_actor.variables, input_train.actor_model.variables, input_train.tau)
    update_target(input_train.target_critic.variables, input_train.critic_model.variables, input_train.tau)

    # save Q value and reward
    input_train.ep_critic_list[input_train.epoch] += q_value
    input_train.ep_reward_list[input_train.epoch] += reward
    rospy.loginfo("reward: %f" % reward)
    rospy.loginfo("q value:%f" % q_value)
    return


def nn_client():
    if iter_current_epoch() is True: return 0

    current_state, current_x, current_y = get_processed_input()

    # try an action
    tf_prev_state = tf.expand_dims(tf.convert_to_tensor(norm(current_state)), 0)
    action = policy_definition(tf_prev_state)
    action_x, action_y = action[0][0], action[0][1]

    # set new state
    new_point = setting_sub_goal(current_x, current_y, action)

    # check if actions are nan
    if np.isnan(action_x) or np.isnan(action_y):
        exit(0)

    rospy.loginfo("Nn output %f,%f", action_x, action_y)
    rospy.loginfo("Current position %f,%f", current_x, current_y)
    rospy.loginfo("New position %f,%f,%f", new_point.x, new_point.y, new_point.theta)

    # send goal and check if it is reached
    try:
        robot_goal_service = rospy.ServiceProxy('robot_goal', NN)
        response = robot_goal_service(new_point)
        new_state, new_x, new_y = get_processed_input()
        save_info_trajectory(input_train.dir_trajectory, new_x, new_y, input_train.epoch,
                             input_train.goal.x, input_train.goal.y, ct.goal_check,
                             response.new_processing)
        if response.new_processing:
            # check if the goal is reached
            if check_goal_reached(new_state) == 0:
                input_train.win_count += 1
                input_train.sequence_goal += 1
                train(new_state, action_x, action_y, current_state, input_train.goal_reward)  # goal reached
                return 2
            train(new_state, action_x, action_y, current_state, 0)  # train
        else:
            # oscillation due to the reset initial position
            if input_train.t < 2:
                rospy.loginfo("Oscillation detected")
                reset_initial_position_random()
            else:
                input_train.collision_count += 1
                train(new_state, action_x, action_y, current_state, input_train.collision_reward)  # collision
                return 0
    except rospy.ServiceException as e:
        rospy.loginfo("Service goal call failed: %s" % e)

    return 1


def ddpg():
    wait_services()
    input_train.bool_norm = 0
    input_train.epoch = 0
    # TODO >> add variant for loading weights
    # load weights
    # in_NN.update_weights_path(dir_weights)
    # loadData(dir_NN_data)
    # loadWeights()
    input_train.result = 0

    # setting the epochs
    for input_train.epoch in range(input_train.epoch, input_train.epochs):
        input_train.ep_reward_list.append(0.0)
        input_train.ep_critic_list.append(0.0)

        rospy.loginfo("epoch %d\n" % input_train.epoch)

        # send goal to initial position for the next epoch
        input_train.t = 0
        if input_train.result == 0 or input_train.sequence_goal > input_train.sequence_goal_max:
            input_train.sequence_goal = 0
            reset_initial_position_random()

        input_train.ou_noise_x = OUActionNoise(mean=np.zeros(1), std_deviation=float(input_train.std_dev) * np.ones(1))
        input_train.ou_noise_y = OUActionNoise(mean=np.zeros(1), std_deviation=float(input_train.std_dev) * np.ones(1))

        goal_inside_occ_grid()

        while True:
            input_train.result = nn_client()
            if input_train.result == 0 or input_train.result == 2:
                break

        input_train.avg_reward_list.append(np.mean(input_train.ep_reward_list[-40:]))
        input_train.avg_reward_list_200.append(np.mean(input_train.ep_reward_list[-200:]))
        input_train.critic_list.append(np.mean(input_train.ep_critic_list[-40:]))
        input_train.critic_list_200.append(np.mean(input_train.ep_critic_list[-200:]))

        # print the rewards
        rospy.loginfo("*******")
        for ep in input_train.ep_reward_list[-40:]: rospy.loginfo("reward %f" % ep)
        rospy.loginfo("*******\n")

        rospy.loginfo("*******")
        for ep in input_train.ep_critic_list[-40:]: rospy.loginfo("q value %f" % ep)
        rospy.loginfo("*******\n")

        rospy.loginfo("*******")
        rospy.loginfo("# goal reached %d" % input_train.win_count)
        rospy.loginfo("# collision %d" % input_train.collision_count)
        rospy.loginfo("*******\n")

        # plot the results
        plot_reward_value(input_train.avg_reward_list, "Rewards mean 40")
        plot_reward_value(input_train.ep_reward_list, "Reward")
        plot_reward_value(input_train.avg_reward_list_200, "Rewards mean 200")
        plot_q_value(input_train.critic_list, "Q value mean 40")
        plot_q_value(input_train.ep_critic_list, "Q value")
        plot_q_value(input_train.critic_list_200, "Q value mean 200")

        save_data_training(dir_NN_data)
        save_weights()


if __name__ == '__main__':
    rospy.init_node('training_DDPG')
    ddpg()
