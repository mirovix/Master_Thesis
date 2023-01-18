#!/usr/bin/env python3

"""
@Author: Miro
@Date: 01/06/2022
@Version: 1.0
@Objective: Config file for define the values for training
@TODO:
"""

import tensorflow as tf
import tensorflow.keras.layers as layers

buffer_size = 50000
batch_size = 64
epochs = 1400
max_t = 30
seq_goal = 20000

std_dev_noise = 0.008
tau_DDPG = 0.001
gamma_DDPG = 0.99

num_states = 14
num_actions = 2

u_bound = 0.45
l_bound = -0.45
min_clip_value = 0.02

critic_lr = 0.001
actor_lr = 0.001

goal_reward = 10
collision_reward = -170
l_max_reward = 0.8
alfa_reward = 70
gamma_reward = 16
k = 0.04
d_infl = 0.75 * l_max_reward

goal_check = 0.216 * 1.6
max_value_norm = 19.84


def get_actor():
    inputs = layers.Input(shape=(num_states,))
    h3 = layers.Dense(600, activation="relu", kernel_initializer=tf.keras.initializers.HeNormal())(inputs)
    h3_ = layers.BatchNormalization()(h3)
    h4 = layers.Dense(600, activation="relu", kernel_initializer=tf.keras.initializers.HeNormal())(h3_)
    h4_ = layers.BatchNormalization()(h4)
    h5 = layers.Dense(600, activation="relu", kernel_initializer=tf.keras.initializers.HeNormal())(h4_)
    h5_ = layers.BatchNormalization()(h5)
    output = layers.Dense(2, activation="tanh",
                          kernel_initializer=tf.random_uniform_initializer(minval=-0.1, maxval=0.1))(h5_)

    output = output * u_bound

    model = tf.keras.Model(inputs, output)

    model.summary()
    return model


def get_critic():
    last_init = tf.random_uniform_initializer(minval=-0.0003, maxval=0.0003)

    # state as input
    state_input = layers.Input(shape=num_states)
    h1_ = layers.Dense(28, activation="relu", kernel_initializer=tf.keras.initializers.HeNormal())(state_input)
    h1 = layers.Dense(56, activation="relu", kernel_initializer=tf.keras.initializers.HeNormal())(h1_)

    # action as input
    action_input = layers.Input(shape=num_actions)
    h2_ = layers.Dense(56, activation="relu", kernel_initializer=tf.keras.initializers.HeNormal())(action_input)

    # both are passed through separate layer before concatenating
    concat = layers.Concatenate()([h1, h2_])

    h2 = layers.Dense(600, activation="relu", kernel_initializer=tf.keras.initializers.HeNormal())(concat)
    h3 = layers.Dense(600, activation="relu", kernel_initializer=tf.keras.initializers.HeNormal())(h2)
    h4 = layers.Dense(600, activation="relu", kernel_initializer=tf.keras.initializers.HeNormal())(h3)
    outputs = layers.Dense(1, activation='linear', kernel_initializer=last_init)(h4)

    # outputs single value for give state-action
    model = tf.keras.Model([state_input, action_input], outputs)

    model.summary()
    return model
