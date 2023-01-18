#!/usr/bin/env python3

"""
@Author: Miro
@Date: 20/05/2022
@Version: 1.1
@Objective: Buffer size for training
@TODO:
"""

import numpy as np
import tensorflow as tf
from DDPG_values import input_train


class Buffer:
    def __init__(self, buffer_capacity=50000, batch_size=64):
        # number of "experiences" to store at max
        self.buffer_capacity = buffer_capacity
        # num of tuples to train on.
        self.batch_size = batch_size

        # its tells us num of times record() was called.
        self.buffer_counter = 0

        # instead of list of tuples as the exp.replay concept go
        # we use different np.arrays for each tuple element
        self.state_buffer = np.zeros((self.buffer_capacity, input_train.num_states))
        self.action_buffer = np.zeros((self.buffer_capacity, input_train.num_actions))
        self.reward_buffer = np.zeros((self.buffer_capacity, 1))
        self.next_state_buffer = np.zeros((self.buffer_capacity, input_train.num_states))

    # Takes (s,a,r,s') observation tuple as input
    def record(self, obs_tuple):
        # set index to zero if buffer_capacity is exceeded,
        # replacing old records
        index = self.buffer_counter % self.buffer_capacity

        self.state_buffer[index] = obs_tuple[0]
        self.action_buffer[index] = obs_tuple[1]
        self.reward_buffer[index] = obs_tuple[2]
        self.next_state_buffer[index] = obs_tuple[3]

        self.buffer_counter += 1

    # eager execution is turned on by default in TensorFlow 2. Decorating with tf.function allows
    # tensorFlow to build a static graph out of the logic and computations in our function.
    # this provides a large speed-up for blocks of code that contain many small TensorFlow operations such as this one.
    @tf.function
    def update(
            self, state_batch, action_batch, reward_batch, next_state_batch
    ):
        # training and updating Actor & Critic networks.
        with tf.GradientTape() as tape:
            target_actions = input_train.target_actor(next_state_batch, training=True)
            y = reward_batch + input_train.gamma * input_train.target_critic(
                [next_state_batch, target_actions], training=True
            )
            critic_value = input_train.critic_model([state_batch, action_batch], training=True)
            critic_loss = tf.math.reduce_mean(tf.math.square(y - critic_value))

        critic_grad = tape.gradient(critic_loss, input_train.critic_model.trainable_variables)
        input_train.critic_optimizer.apply_gradients(
            zip(critic_grad, input_train.critic_model.trainable_variables)
        )

        with tf.GradientTape() as tape:
            actions = input_train.actor_model(state_batch, training=True)
            critic_value = input_train.critic_model([state_batch, actions], training=True)
            # used `-value` as we want to maximize the value given
            # by the critic for our actions
            actor_loss = -tf.math.reduce_mean(critic_value)

        actor_grad = tape.gradient(actor_loss, input_train.actor_model.trainable_variables)
        input_train.actor_optimizer.apply_gradients(
            zip(actor_grad, input_train.actor_model.trainable_variables)
        )

    # we compute the loss and update parameters
    def learn(self):
        # get sampling range
        record_range = min(self.buffer_counter, self.buffer_capacity)
        # randomly sample indices
        batch_indices = np.random.choice(record_range, self.batch_size)

        # convert to tensors
        state_batch = tf.convert_to_tensor(self.state_buffer[batch_indices])
        action_batch = tf.convert_to_tensor(self.action_buffer[batch_indices])
        reward_batch = tf.convert_to_tensor(self.reward_buffer[batch_indices])
        reward_batch = tf.cast(reward_batch, dtype=tf.float32)
        next_state_batch = tf.convert_to_tensor(self.next_state_buffer[batch_indices])

        self.update(state_batch, action_batch, reward_batch, next_state_batch)
