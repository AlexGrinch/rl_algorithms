import os
import time
import random
import numpy as np

import tensorflow as tf
import tensorflow.contrib.layers as layers

import gym
from gym import spaces
from PIL import Image
from collections import deque, namedtuple
from atari_wrappers import wrap_deepmind

from IPython import display
import matplotlib.pyplot as plt

class QNetwork:
    
    def __init__(self, num_actions, state_shape=[84, 84, 4],
                 convs=[[32, 8, 4], [64, 4, 2], [64, 3, 1]], 
                 fully_connected=[512], architecture="DQN", 
                 num_atoms=21, v=(-10, 10),
                 optimizer = tf.train.AdamOptimizer(2.5e-4),
                 scope="q_network", reuse=False):
        """Class for neural network which estimates Q-function
        
        Parameters
        ----------
        num_actions: int
            number of actions the agent can take
        state_shape: list
            list of 3 parameters [frame_w, frame_h, num_frames]
            frame_w: frame width
            frame_h: frame height
            num_frames: number of successive frames treated as a state
        convs: list
            list of convolutional layers' parameters of the form
            [[num_outputs, kernel_size, stride], ...]
        fully_connected: list
            list of fully connected layers' parameters of the form
            [num_outputs, ...]
        architecture: str
            architecture type, possible values
            "DQN" -- Deep Q-Network (https://arxiv.org/abs/1312.5602)
            "DuelDQN" -- Dueling DQN (https://arxiv.org/abs/1511.06581)
            "DistDQN" -- Distributional DQN (https://arxiv.org/abs/1707.06887)
        num_atoms: int 
            number of atoms in distribution support (DistDQN only)
        v: tuple
            tuple of 2 parameters (v_min, v_max) (DistDQN only)
            v_min: minimum q-function value
            v_max: maximum q-function value    
        optimizer: tf.train optimizer
            optimization algorithm for stochastic gradient descend
        scope: str
            unique name of a specific network
        reuse: bool
            tensorflow standard parameter
        """
        
        xavier = layers.xavier_initializer()
        
        ###################### Neural network architecture ######################
        
        input_shape = [None] + state_shape
        self.input_states = tf.placeholder(dtype=tf.float32, shape=input_shape)

        with tf.variable_scope(scope, reuse=reuse):
            # convolutional part of the network
            conv_layers = [self.input_states]
            with tf.variable_scope("conv"):
                for num_outputs, kernel_size, stride in convs:
                    conv = layers.convolution2d(conv_layers[-1], 
                                                num_outputs=num_outputs,
                                                kernel_size=kernel_size,
                                                stride=stride,
                                                padding='VALID',
                                                biases_initializer=None,
                                                activation_fn=tf.nn.relu)
                    conv_layers.append(conv)
                self.conv_layers = conv_layers[1:]
            self.conv_out = layers.flatten(self.conv_layers[-1])

            # fully connected part of the network
            fc_layers = [self.conv_out]
            with tf.variable_scope("fc"):
                for num_outputs in fully_connected:
                    fc = layers.fully_connected(fc_layers[-1],
                                                num_outputs=num_outputs,
                                                activation_fn=tf.nn.relu,
                                                biases_initializer=None,
                                                weights_initializer=xavier)
                    fc_layers.append(fc)
                self.fc_layers = fc_layers[1:]
            self.fc_out = self.fc_layers[-1]
            
            if architecture == "DQN":
                
                with tf.variable_scope("q_values"):
                    q_weights = tf.Variable(xavier([fully_connected[-1], num_actions]))
                    self.q_state_values = tf.matmul(self.fc_out, q_weights)
                    
            elif architecture == "DistDQN":
                
                # discrete distribution parameters
                self.num_atoms = num_atoms
                self.v_min, self.v_max = v
                self.delta_z = (self.v_max - self.v_min) / (self.num_atoms - 1)
                self.z = [self.v_min + i * self.delta_z for i in range(self.num_atoms)]   
                
                # probabilities of atoms
                self.action_probs = []
                with tf.variable_scope("probs"):
                    for a in range(num_actions):
                        action_prob = layers.fully_connected(self.fc_out,
                                                              num_outputs=self.num_atoms,
                                                              activation_fn=None,
                                                              biases_initializer=None,
                                                              weights_initializer=xavier)
                        action_prob = tf.nn.softmax(action_prob)
                        self.action_probs.append(action_prob)
                    self.state_probs = tf.stack(self.action_probs, axis=1)
                    
                # q-values as expectations
                with tf.variable_scope("q_values"):
                    q_values = []
                    for a in range(num_actions):
                        q_value = tf.reduce_sum(self.z * self.action_probs[a], axis=1)
                        q_value = tf.reshape(q_value, [-1, 1])
                        q_values.append(q_value)
                    self.q_state_values = tf.concat(q_values, axis=1)
                                                         
        ######################### Optimization procedure ########################
        
        # Q-function approximation
        self.input_actions = tf.placeholder(dtype=tf.int32, shape=[None])
        actions_onehot = tf.one_hot(self.input_actions, num_actions, dtype=tf.float32)
        actions_onehot_reshaped = tf.reshape(actions_onehot, [-1, num_actions, 1])
        
        q_values_selected = tf.multiply(self.q_state_values, actions_onehot)
        self.q_action_values = tf.reduce_sum(q_values_selected, axis=1)
        self.q_max_values = tf.argmax(self.q_state_values, axis=1)
        
        if architecture == "DQN":
            self.input_target = tf.placeholder(dtype=tf.float32, shape=[None])
            self.td_error = tf.losses.huber_loss(self.input_target, self.q_action_values)
            self.loss = tf.reduce_mean(self.td_error)
            
        elif architecture == "DistDQN":
            self.input_target = tf.placeholder(dtype=tf.float32, shape=[None, self.num_atoms])
            probs_selected = tf.multiply(self.state_probs, actions_onehot_reshaped)
            self.probs = tf.reduce_sum(probs_selected, axis=1)
            self.loss = -tf.reduce_sum(self.input_target * tf.log(self.probs))

        self.optimizer = optimizer
        self.update_model = self.optimizer.minimize(self.loss)
        
    
    def cat_proj(self, sess, rewards, states_, actions_, end, gamma=0.99):
        """
        Categorical algorithm from https://arxiv.org/abs/1707.06887
        """
        
        feed_dict = {self.input_states:states_, self.input_actions:actions_}
        probs = sess.run(self.probs, feed_dict=feed_dict)
        m = np.zeros_like(probs)
        rewards = np.array(rewards, dtype=np.float32)
        end = np.array(end, dtype=np.float32)
        batch_size = rewards.size
        
        for j in range(self.num_atoms):
            Tz = rewards + gamma * end * self.z[j]
            Tz = np.minimum(self.v_max, np.maximum(self.v_min, Tz))
            b = (Tz - self.v_min) / self.delta_z
            l = np.floor(b)
            u = np.ceil(b)
            m[np.arange(batch_size), l.astype(int)] += probs[:,j] * (u - b)
            m[np.arange(batch_size), u.astype(int)] += probs[:,j] * (b - l) 
            
        return m 
            
    
    def argmax_q(self, sess, states):
        feed_dict = {self.input_states:states}
        q_max_values = sess.run(self.q_max_values, feed_dict)
        return q_max_values

    def q_values(self, sess, states):
        feed_dict = {self.input_states:states}
        q_state_values = sess.run(self.q_state_values, feed_dict)
        return q_state_values

    def update(self, sess, states, actions, input_target):
        feed_dict = {self.input_states:states,
                     self.input_actions:actions,
                     self.input_target:input_target}
        sess.run(self.update_model, feed_dict)
    
    
class ReplayMemory:

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0
        self.transition = namedtuple('Transition', 
                                     ('s', 'a', 'r', 's_', 'end'))

    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = [*args]
        self.position = (self.position + 1) % self.capacity
    
    def get_batch(self, batch_size):
        batch = random.sample(self.memory, batch_size)
        batch = np.reshape(batch, [batch_size, 5])
        s = np.stack(batch[:,0])
        a = batch[:,1]
        r = batch[:,2]
        s_ = np.stack(batch[:,3])
        end = 1 - batch[:,4]
        return self.transition(s, a, r, s_, end)

    def __len__(self):
        return len(self.memory)