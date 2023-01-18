#!/usr/bin/env python3

"""
@Author: Miro
@Date: 01/06/2022
@Version: 1.0
@Objective: Input values for training model
@TODO:
"""

import os
import tensorflow as tf
from training import config_train as ct
from geometry_msgs.msg import Point


class InputValuesTrain:
    def __init__(self):
        self.l_max = ct.l_max_reward
        self.num_states = ct.num_states
        self.num_actions = ct.num_actions
        self.upper_bound = ct.u_bound
        self.lower_bound = ct.l_bound

        self.critic_lr = ct.critic_lr
        self.actor_lr = ct.actor_lr
        self.critic_optimizer = tf.keras.optimizers.Adam(ct.critic_lr)
        self.actor_optimizer = tf.keras.optimizers.Adam(ct.actor_lr)

        self.actor_model = None
        self.critic_model = None
        self.target_actor = None
        self.target_critic = None
        self.gamma = ct.gamma_DDPG
        self.tau = ct.tau_DDPG

        self.result = None

        self.goal_policy = None
        self.exit_policy = 0
        self.count_input = 0

        self.win_count = 0
        self.collision_count = 0

        self.prev_state = None
        self.previous_shaping = None
        self.goal_reward = ct.goal_reward
        self.collision_reward = ct.collision_reward
        self.alfa_reward = ct.alfa_reward
        self.gamma_reward = ct.gamma_reward
        self.k = ct.k
        self.d_infl = ct.d_infl
        self.max_value_norm = ct.max_value_norm
        self.bool_norm = 0

        self.std_dev = ct.std_dev_noise
        self.ou_noise_x = None
        self.ou_noise_y = None
        self.buffer = None
        self.buffer_size = ct.buffer_size
        self.batch_size = ct.batch_size

        self.RL_pub = None

        self.avg_reward_list = []
        self.ep_reward_list = []
        self.ep_critic_list = []
        self.ep_loss_list = []

        self.avg_reward_list = []
        self.avg_reward_list_200 = []
        self.critic_list = []
        self.critic_list_200 = []

        self.dir_plot = None
        self.dir_weights = None
        self.dir_trajectory = None

        self.init_occ = None

        self.epochs = ct.epochs
        self.epoch = 0
        self.max_t = ct.max_t
        self.t = 0
        self.load = 1
        self.sequence_goal = 0
        self.sequence_goal_max = ct.seq_goal

        self.goal = Point()

    def reset_list(self):
        self.avg_reward_list = []
        self.ep_reward_list = []
        self.ep_critic_list = []
        self.ep_loss_list = []

    def update_prev_state(self, new_prev_state):
        self.prev_state = new_prev_state

    def update_prev_shaping(self, new_prev_shaping):
        self.previous_shaping = new_prev_shaping

    def update_action_noise_buffer(self, noise_x, noise_y, buffer):
        self.ou_noise_x = noise_x
        self.ou_noise_y = noise_y
        self.buffer = buffer

    def update_actor_critic_model(self):
        self.actor_model = ct.get_actor()
        self.critic_model = ct.get_critic()
        self.target_actor = ct.get_actor()
        self.target_critic = ct.get_critic()
        self.target_actor.set_weights(self.actor_model.get_weights())
        self.target_critic.set_weights(self.critic_model.get_weights())

    def update_plot_path(self, dir_name):
        self.dir_plot = dir_name
        os.chdir(os.path.dirname(dir_name))

    def update_weights_path(self, dir_name):
        self.dir_weights = dir_name

    def update_trajectory_path(self, dir_name):
        self.dir_trajectory = dir_name
