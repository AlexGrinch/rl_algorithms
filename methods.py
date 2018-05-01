import numpy as np
import random
import tensorflow as tf
import tensorflow.contrib.layers as layers
from tensorflow.contrib.layers import convolution2d as conv
from tensorflow.contrib.layers import fully_connected as fc
from tensorflow.contrib.layers import xavier_initializer as xavier

####################################################################################################
########################################### Core modules ###########################################
####################################################################################################

def conv_module(input_layer, convs, activation_fn=tf.nn.relu):
    """ convolutional module
    """
    out = input_layer
    for num_outputs, kernel_size, stride in convs:
        out = conv(out,
                   num_outputs=num_outputs,
                   kernel_size=kernel_size,
                   stride=stride,
                   padding='VALID',
                   activation_fn=activation_fn)
    return out
    
def fc_module(input_layer, fully_connected, activation_fn=tf.nn.relu):
    """ fully connected module
    """
    out = input_layer
    for num_outputs in fully_connected:
        out = fc(out,
                 num_outputs=num_outputs,
                 activation_fn=activation_fn,
                 weights_initializer=xavier())
    return out

def func_module(input_layer, num_inputs, num_outputs):
    """ final module which estimates some function (value, q, policy, etc)
    """
    out = input_layer
    out_weights = tf.Variable(xavier()([num_inputs, num_outputs]))
    out = tf.matmul(out, out_weights)
    return out

def full_module(input_layer, convs, fully_connected, num_outputs, 
                        activation_fn=tf.nn.relu):
    """ convolutional + fully connected + functional module
    """
    out = input_layer
    out = conv_module(out, convs, activation_fn)
    out = layers.flatten(out)
    out = fc_module(out, fully_connected, activation_fn)
    out = func_module(out, fully_connected[-1], num_outputs)
    return out

####################################################################################################
########################################## Deep Q-Network ##########################################
####################################################################################################

class DeepQNetwork:

    def __init__(self, num_actions, state_shape=[8, 8, 5],
                 convs=[[32, 4, 2], [64, 2, 1]], 
                 fully_connected=[128],
                 activation_fn=tf.nn.relu,
                 optimizer=tf.train.AdamOptimizer(2.5e-4, epsilon=0.01/32),
                 gradient_clip=10.0,
                 scope="dqn", reuse=False):
        
        with tf.variable_scope(scope, reuse=reuse):

            ###################### Neural network architecture ######################

            input_shape = [None] + state_shape
            self.input_states = tf.placeholder(dtype=tf.float32, shape=input_shape)

            self.q_values = full_module(self.input_states, convs, fully_connected,
                                        num_actions, activation_fn)

            ######################### Optimization procedure ########################

            # convert input actions to indices for q-values selection
            self.input_actions = tf.placeholder(dtype=tf.int32, shape=[None])
            indices_range = tf.range(tf.shape(self.input_actions)[0])
            action_indices = tf.stack([indices_range, self.input_actions], axis=1)
            
            # select q-values for input actions
            self.q_values_selected = tf.gather_nd(self.q_values, action_indices)
            
            # select best actions (according to q-values)
            self.q_argmax = tf.argmax(self.q_values, axis=1)

            # define loss function and update rule
            self.q_targets = tf.placeholder(dtype=tf.float32, shape=[None])
            self.loss = tf.losses.huber_loss(self.q_targets, self.q_values_selected, 
                                             delta=gradient_clip)
            self.update_model = optimizer.minimize(self.loss)

    def get_q_values_s(self, sess, states):
        feed_dict = {self.input_states:states}
        q_values = sess.run(self.q_values, feed_dict)
        return q_values
    
    def get_q_values_sa(self, sess, states, actions):
        feed_dict = {self.input_states:states, self.input_actions:actions}
        q_values_selected = sess.run(self.q_values_selected, feed_dict)
        return q_values_selected
    
    def get_q_argmax(self, sess, states):
        feed_dict = {self.input_states:states}
        q_argmax = sess.run(self.q_argmax, feed_dict)
        return q_argmax

    def update(self, sess, states, actions, q_targets):

        feed_dict = {self.input_states:states,
                     self.input_actions:actions,
                     self.q_targets:q_targets}
        sess.run(self.update_model, feed_dict)
        
####################################################################################################
###################################### Dueling Deep Q-Network ######################################
#################################################################################################### 

class DuelingDeepQNetwork:
    
    def __init__(self, num_actions, state_shape=[8, 8, 5],
                 convs=[[32, 4, 2], [64, 2, 1]],
                 fully_connected=[64],
                 activation_fn=tf.nn.relu,
                 optimizer=tf.train.AdamOptimizer(2.5e-4, epsilon=0.01/32),
                 gradient_clip=10.0,
                 scope="duel_dqn", reuse=False):

        with tf.variable_scope(scope, reuse=reuse):
            
            ###################### Neural network architecture ######################
            
            input_shape = [None] + state_shape
            self.input_states = tf.placeholder(dtype=tf.float32, shape=input_shape)
            
            out = conv_module(self.input_states, convs, activation_fn)
            val, adv = tf.split(out, num_or_size_splits=2, axis=3)
            self.v_values = full_module(val, [], fully_connected, 1, activation_fn)
            self.a_values = full_module(adv, [], fully_connected, num_actions, activation_fn)
                
            a_values_mean = tf.reduce_mean(self.a_values, axis=1, keepdims=True)
            a_values_centered = tf.subtract(self.a_values, a_values_mean)
            self.q_values = self.v_values + a_values_centered

            ######################### Optimization procedure ########################

            # convert input actions to indices for q-values selection
            self.input_actions = tf.placeholder(dtype=tf.int32, shape=[None])
            indices_range = tf.range(tf.shape(self.input_actions)[0])
            action_indices = tf.stack([indices_range, self.input_actions], axis=1)
            
            # select q-values for input actions
            self.q_values_selected = tf.gather_nd(self.q_values, action_indices)
            
            # select best actions (according to q-values)
            self.q_argmax = tf.argmax(self.q_values, axis=1)

            # define loss function and update rule
            self.q_targets = tf.placeholder(dtype=tf.float32, shape=[None])
            self.loss = tf.losses.huber_loss(self.q_targets, self.q_values_selected,
                                             delta=gradient_clip)
            self.update_model = optimizer.minimize(self.loss)
            
    def get_q_values_s(self, sess, states):
        feed_dict = {self.input_states:states}
        q_values = sess.run(self.q_values, feed_dict)
        return q_values
    
    def get_q_values_sa(self, sess, states, actions):
        feed_dict = {self.input_states:states, self.input_actions:actions}
        q_values_selected = sess.run(self.q_values_selected, feed_dict)
        return q_values_selected
    
    def get_q_argmax(self, sess, states):
        feed_dict = {self.input_states:states}
        q_argmax = sess.run(self.q_argmax, feed_dict)
        return q_argmax

    def update(self, sess, states, actions, q_targets):

        feed_dict = {self.input_states:states,
                     self.input_actions:actions,
                     self.q_targets:q_targets}
        sess.run(self.update_model, feed_dict)
        
####################################################################################################
#################################### Categorical Deep Q-Network ####################################
####################################################################################################

class CategoricalDeepQNetwork:
    
    def __init__(self, num_actions, state_shape=[8, 8, 5],
                 convs=[[32, 4, 2], [64, 2, 1]],
                 fully_connected=[128], 
                 num_atoms=21, v=(-10, 10),
                 activation_fn=tf.nn.relu,
                 optimizer=tf.train.AdamOptimizer(2.5e-4, epsilon=0.01/32),
                 scope="cat_dqn", reuse=False):
        
        with tf.variable_scope(scope, reuse=reuse):

            ###################### Neural network architecture ######################

            input_shape = [None] + state_shape
            self.input_states = tf.placeholder(dtype=tf.float32, shape=input_shape)
        
            # distribution parameters
            self.num_atoms = num_atoms
            self.v_min, self.v_max = v
            self.delta_z = (self.v_max - self.v_min) / (self.num_atoms - 1)
            self.z = [self.v_min + i * self.delta_z for i in range(self.num_atoms)]
            self.tensor_z = tf.convert_to_tensor(self.z, dtype=tf.float32)

            out = conv_module(self.input_states, convs, activation_fn)
            out = layers.flatten(out)
            out = fc_module(out, fully_connected, activation_fn)
            out = fc_module(out, [num_actions * self.num_atoms], None)
            self.logits = tf.reshape(out, shape=[-1, num_actions, self.num_atoms])
            self.probs = tf.nn.softmax(self.logits, axis=2)
            self.q_values = tf.reduce_sum(tf.multiply(self.probs, self.tensor_z), axis=2)

            ######################### Optimization procedure ########################

            # convert input actions to indices for probs and q-values selection
            self.input_actions = tf.placeholder(dtype=tf.int32, shape=[None])
            indices_range = tf.range(tf.shape(self.input_actions)[0])
            action_indices = tf.stack([indices_range, self.input_actions], axis=1)

            # select q-values and probs for input actions
            self.q_values_selected = tf.gather_nd(self.q_values, action_indices)
            self.logits_selected = tf.gather_nd(self.logits, action_indices)
            self.probs_selected = tf.gather_nd(self.probs, action_indices)
            
            # select best actions (according to q-values)
            self.q_argmax = tf.argmax(self.q_values, axis=1)

            # define loss function and update rule
            self.probs_targets = tf.placeholder(dtype=tf.float32, shape=[None, self.num_atoms])
            self.loss = -tf.reduce_sum(self.probs_targets * self.logits_selected)
            self.update_model = optimizer.minimize(self.loss)
    
    def get_q_values_s(self, sess, states):
        feed_dict = {self.input_states:states}
        q_values = sess.run(self.q_values, feed_dict)
        return q_values
    
    def get_q_values_sa(self, sess, states, actions):
        feed_dict = {self.input_states:states, self.input_actions:actions}
        q_values_selected = sess.run(self.q_values_selected, feed_dict)
        return q_values_selected
    
    def get_q_argmax(self, sess, states):
        feed_dict = {self.input_states:states}
        q_argmax = sess.run(self.q_argmax, feed_dict)
        return q_argmax
    
    def get_probs_s(self, sess, states):
        feed_dict = {self.input_states:states}
        probs = sess.run(self.probs, feed_dict)
        return probs
    
    def get_probs_sa(self, sess, states, actions):
        feed_dict = {self.input_states:states, self.input_actions:actions}
        probs_selected = sess.run(self.probs_selected, feed_dict)
        return probs_selected
    
    def update(self, sess, states, actions, probs_targets):

        feed_dict = {self.input_states:states,
                     self.input_actions:actions,
                     self.probs_targets:probs_targets}
        sess.run(self.update_model, feed_dict)

    def cat_proj(self, sess, rewards, states_, actions_, done, gamma=0.99):
        """
        Categorical algorithm from https://arxiv.org/abs/1707.06887
        """

        probs = self.get_probs_sa(sess, states_, actions_)
        m = np.zeros_like(probs)
        rewards = np.array(rewards, dtype=np.float32)
        done = np.array(done, dtype=np.float32)
        batch_size = rewards.size

        for j in range(self.num_atoms):
            Tz = rewards + gamma * (1 - done) * self.z[j]
            Tz = np.minimum(self.v_max, np.maximum(self.v_min, Tz))
            b = (Tz - self.v_min) / self.delta_z
            l = np.floor(b)
            u = np.ceil(b)
            m[np.arange(batch_size), l.astype(int)] += probs[:,j] * (u - b)
            m[np.arange(batch_size), u.astype(int)] += probs[:,j] * (b - l)

        return m
        
####################################################################################################
################################ Qantile Regression Deep Q-Network #################################
####################################################################################################

class QuantileRegressionDeepQNetwork:
    
    def __init__(self, num_actions, state_shape=[8, 8, 5],
                 convs=[[32, 4, 2], [64, 2, 1]],
                 fully_connected=[128], 
                 num_atoms=50, kappa=1.0,
                 activation_fn=tf.nn.relu,
                 optimizer=tf.train.AdamOptimizer(2.5e-4, epsilon=0.01/32),
                 scope="qr_dqn", reuse=False):
        
        with tf.variable_scope(scope, reuse=reuse):

            ###################### Neural network architecture ######################

            input_shape = [None] + state_shape
            self.input_states = tf.placeholder(dtype=tf.float32, shape=input_shape)
        
            # distribution parameters
            tau_min = 1 / (2 * num_atoms) 
            tau_max = 1 - tau_min
            tau_vector = tf.lin_space(start=tau_min, stop=tau_max, num=num_atoms)
            
            # reshape tau to matrix for fast loss calculation
            tau_matrix = tf.tile(tau_vector, [num_atoms])
            self.tau_matrix = tf.reshape(tau_matrix, shape=[num_atoms, num_atoms])
            
            # main module
            out = full_module(self.input_states, convs, fully_connected,
                              num_outputs=num_actions*num_atoms, activation_fn=activation_fn)
            self.atoms = tf.reshape(out, shape=[-1, num_actions, num_atoms])
            self.q_values = tf.reduce_mean(self.atoms, axis=2)

            ######################### Optimization procedure ########################

            # convert input actions to indices for atoms and q-values selection
            self.input_actions = tf.placeholder(dtype=tf.int32, shape=[None])
            indices_range = tf.range(tf.shape(self.input_actions)[0])
            action_indices = tf.stack([indices_range, self.input_actions], axis=1)

            # select q-values for input actions
            self.q_values_selected = tf.gather_nd(self.q_values, action_indices)
            self.atoms_selected = tf.gather_nd(self.atoms, action_indices)
            
            # select best actions (according to q-values)
            self.q_argmax = tf.argmax(self.q_values, axis=1)
            
            # reshape chosen atoms to matrix for fast loss calculation
            atoms_matrix = tf.tile(self.atoms_selected, [1, num_atoms])
            self.atoms_matrix = tf.reshape(atoms_matrix, shape=[-1, num_atoms, num_atoms])
            
            # reshape target atoms to matrix for fast loss calculation
            self.atoms_targets = tf.placeholder(dtype=tf.float32, shape=[None, num_atoms])
            targets_matrix = tf.tile(self.atoms_targets, [1, num_atoms])
            targets_matrix = tf.reshape(targets_matrix, shape=[-1, num_atoms, num_atoms])
            self.targets_matrix = tf.transpose(targets_matrix, perm=[0, 2, 1])
            
            # define loss function and update rule
            atoms_diff = self.targets_matrix - self.atoms_matrix
            delta_atoms_diff = tf.where(atoms_diff<0, tf.zeros_like(atoms_diff), tf.ones_like(atoms_diff))
            huber_weights = tf.abs(self.tau_matrix - delta_atoms_diff) / num_atoms
            self.loss = tf.losses.huber_loss(self.targets_matrix, self.atoms_matrix, weights=huber_weights,
                                             delta=kappa, reduction=tf.losses.Reduction.SUM)
            self.update_model = optimizer.minimize(self.loss)

    def get_q_values_s(self, sess, states):
        feed_dict = {self.input_states:states}
        q_values = sess.run(self.q_values, feed_dict)
        return q_values
    
    def get_q_values_sa(self, sess, states, actions):
        feed_dict = {self.input_states:states, self.input_actions:actions}
        q_values_selected = sess.run(self.q_values_selected, feed_dict)
        return q_values_selected
    
    def get_q_argmax(self, sess, states):
        feed_dict = {self.input_states:states}
        q_argmax = sess.run(self.q_argmax, feed_dict)
        return q_argmax
    
    def get_atoms_s(self, sess, states):
        feed_dict = {self.input_states:states}
        atoms = sess.run(self.atoms, feed_dict)
        return probs
    
    def get_atoms_sa(self, sess, states, actions):
        feed_dict = {self.input_states:states, self.input_actions:actions}
        atoms_selected = sess.run(self.atoms_selected, feed_dict)
        return atoms_selected

    def update(self, sess, states, actions, atoms_targets):

        feed_dict = {self.input_states:states,
                     self.input_actions:actions,
                     self.atoms_targets:atoms_targets}
        sess.run(self.update_model, feed_dict)
        
####################################################################################################
######################################## Soft Actor-Critic #########################################
####################################################################################################

class SoftActorCriticNetwork:

    def __init__(self, num_actions, state_shape=[8, 8, 5],
                 convs=[[32, 4, 2], [64, 2, 1]], 
                 fully_connected=[128],
                 activation_fn=tf.nn.relu,
                 optimizers=[tf.train.AdamOptimizer(2.5e-4),
                             tf.train.AdamOptimizer(2.5e-4),
                             tf.train.AdamOptimizer(2.5e-4)],
                 scope="sac", reuse=False):
        
        with tf.variable_scope(scope, reuse=reuse):
        
            ###################### Neural network architecture ######################

            input_shape = [None] + state_shape
            self.input_states = tf.placeholder(dtype=tf.float32, shape=input_shape)

            self.v_values = full_module(self.input_states, convs, fully_connected, 
                                        1, activation_fn)
            self.q_values = full_module(self.input_states, convs, fully_connected, 
                                        num_actions, activation_fn)
            self.p_logits = full_module(self.input_states, convs, fully_connected,
                                        num_actions, activation_fn)
            self.p_values = layers.softmax(self.p_logits)

            ######################### Optimization procedure ########################

            # convert input actions to indices for p-logits and q-values selection
            self.input_actions = tf.placeholder(dtype=tf.int32, shape=[None])
            indices_range = tf.range(tf.shape(self.input_actions)[0])
            action_indices = tf.stack([indices_range, self.input_actions], axis=1)
            
            q_values_selected = tf.gather_nd(self.q_values, action_indices)
            p_logits_selected = tf.gather_nd(self.p_logits, action_indices)

            # choose best actions (according to q-values)
            self.q_argmax = tf.argmax(self.q_values, axis=1)

            # define loss function and update rule
            self.q_targets = tf.placeholder(dtype=tf.float32, shape=[None])
            self.v_targets = tf.placeholder(dtype=tf.float32, shape=[None])
            self.p_targets = tf.placeholder(dtype=tf.float32, shape=[None])

            q_loss = tf.losses.huber_loss(self.q_targets, q_values_selected)
            self.q_loss = tf.reduce_sum(q_loss)
            q_optimizer = optimizers[0]

            v_loss = tf.losses.huber_loss(self.v_targets[:,None], self.v_values)
            self.v_loss = tf.reduce_sum(v_loss)
            v_optimizer = optimizers[1]

            p_loss = tf.losses.huber_loss(self.p_targets, p_logits_selected)
            self.p_loss = tf.reduce_sum(p_loss)
            p_optimizer = optimizers[2]

            self.update_q_values = q_optimizer.minimize(self.q_loss)
            self.update_v_values = v_optimizer.minimize(self.v_loss)
            self.update_p_logits = p_optimizer.minimize(self.p_loss)

    def get_q_argmax(self, sess, states):
        feed_dict = {self.input_states:states}
        q_argmax = sess.run(self.q_argmax, feed_dict)
        return q_argmax

    def get_q_values_s(self, sess, states):
        feed_dict = {self.input_states:states}
        q_values = sess.run(self.q_values, feed_dict)
        return q_values
    
    def get_v_values_s(self, sess, states):
        feed_dict = {self.input_states:states}
        v_values = sess.run(self.v_values, feed_dict)
        return v_values
    
    def get_p_logits_s(self, sess, states):
        feed_dict = {self.input_states:states}
        p_logits = sess.run(self.p_logits, feed_dict)
        return p_logits
    
    def get_p_values_s(self, sess, states):
        feed_dict = {self.input_states:states}
        p_values = sess.run(self.p_values, feed_dict)
        return p_values

    def update_q(self, sess, states, actions, q_targets):
        
        feed_dict = {self.input_states:states,
                     self.input_actions:actions,
                     self.q_targets:q_targets}
        sess.run(self.update_q_values, feed_dict)
        
    def update_v(self, sess, states, v_targets):
        
        feed_dict = {self.input_states:states,
                     self.v_targets:v_targets}
        sess.run(self.update_v_values, feed_dict)
        
    def update_p(self, sess, states, actions, p_targets):
        
        feed_dict = {self.input_states:states,
                     self.input_actions:actions,
                     self.p_targets:p_targets}
        sess.run(self.update_p_logits, feed_dict) 