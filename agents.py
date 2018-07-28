import os
import time

from IPython import display
import matplotlib.pyplot as plt

from methods import *
from utils import *

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

class Agent:
    
    def __init__(self, env, num_actions, state_shape=[8, 8, 5], 
                 save_path="rl_models", model_name="agent"):
        
        self.train_env = env
        self.num_actions = num_actions
            
        self.path = save_path + "/" + model_name
        if not os.path.exists(self.path):
            os.makedirs(self.path)
            
    def init_weights(self):
        
        global_vars = tf.global_variables(scope="agent")+tf.global_variables(scope="target")
        self.init = tf.variables_initializer(global_vars)
        self.saver = tf.train.Saver()
        
        self.agent_vars = tf.trainable_variables(scope="agent")
        self.target_vars = tf.trainable_variables(scope="target")
        
    def set_parameters(self, 
                       replay_memory_size=50000,
                       replay_start_size=10000,
                       init_eps=1,
                       final_eps=0.02,
                       annealing_steps=100000,
                       discount_factor=0.99,
                       max_episode_length=2000,
                       frame_history_len=1):
        
        # create experience replay and fill it with random policy samples
        self.rep_buffer = ReplayBuffer(size=replay_memory_size, 
                                       frame_history_len=frame_history_len)
        frame_count = 0
        while (frame_count < replay_start_size):
            last_obs = self.train_env.reset()
            for time_step in range(max_episode_length):
                
                last_idx = self.rep_buffer.store_frame(last_obs)
                recent_obs = self.rep_buffer.encode_recent_observation()
                action = np.random.randint(self.num_actions)
                obs, reward, done = self.train_env.step(action)[:3]
                self.rep_buffer.store_effect(last_idx, action, reward, done)

                frame_count += 1
                
                if done: break    
                last_obs = obs
        
        # define epsilon decrement schedule for exploration
        self.eps = init_eps
        self.final_eps = final_eps
        self.eps_drop = (init_eps - final_eps) / annealing_steps
        
        self.gamma = discount_factor
        self.max_ep_length = max_episode_length
        
    def train(self,
              gpu_id=0,
              batch_size=32,
              exploration="e-greedy",
              agent_update_freq=4,
              target_update_freq=5000,
              tau=1,
              max_num_epochs=50000,
              performance_print_freq=500,
              save_freq=10000, 
              from_epoch=0):
        
        config = self.gpu_config(gpu_id)
        target_ops = self.update_target_graph(tau)
        self.batch_size = batch_size
        
        with tf.Session(config=config) as sess:
            
            if from_epoch == 0:
                sess.run(self.init)
                train_rewards = []
                frame_counts = []
                frame_count = 0
                num_epochs = 0
                
            else:
                self.saver.restore(sess, self.path+"/model-"+str(from_epoch))
                train_rewards = list(np.load(self.path+"/learning_curve.npz")["r"])
                frame_counts = list(np.load(self.path+"/learning_curve.npz")["f"])
                frame_count = frame_counts[-1]
                num_epochs = from_epoch
    
            episode_count = 0
            ep_lifetimes = []
        
            
            while num_epochs < max_num_epochs:
                
                train_ep_reward = 0
                
                # reset the environment / start new game
                last_obs = self.train_env.reset()
                for time_step in range(self.max_ep_length):
                    
                    last_idx = self.rep_buffer.store_frame(last_obs)
                    recent_obs = self.rep_buffer.encode_recent_observation()
                    
                    # choose action e-greedily
                    action = self.choose_action(sess, recent_obs, exploration)
                        
                    # make step in the environment    
                    obs, reward, done = self.train_env.step(action)[:3]
                                        
                    # save transition into experience replay
                    self.rep_buffer.store_effect(last_idx, action, reward, done)
                    
                    # update current state and statistics
                    frame_count += 1
                    train_ep_reward += reward
                    
                    # reduce epsilon according to schedule
                    if self.eps > self.final_eps:
                        self.eps -= self.eps_drop
                    
                    # update network weights
                    if frame_count % agent_update_freq == 0:
                        
                        batch = self.rep_buffer.sample(batch_size)
                        self.update_agent_weights(sess, batch)
                        
                        # update target network
                        if tau == 1:
                            if frame_count % target_update_freq == 0:
                                self.update_target_weights(sess, target_ops)
                        else: self.update_target_weights(sess, target_ops)
                    
                    # make checkpoints of network weights and save learning curve
                    if frame_count % save_freq == 1:
                        num_epochs += 1
                        try:
                            self.saver.save(sess, self.path+"/model", global_step=num_epochs)
                            np.savez(self.path+"/learning_curve.npz", r=train_rewards, 
                                     f=frame_counts, l=ep_lifetimes)
                        except: pass
                    
                    # if game is over, reset the environment
                    if done: break
                    last_obs = obs

                episode_count += 1
                train_rewards.append(train_ep_reward)
                frame_counts.append(frame_count)
                ep_lifetimes.append(time_step+1)
                
                # print performance once in a while
                if episode_count % performance_print_freq == 0:
                    avg_reward = np.mean(train_rewards[-performance_print_freq:])
                    avg_lifetime = np.mean(ep_lifetimes[-performance_print_freq:])
                    print("frame count:", frame_count)
                    print("average reward:", avg_reward)
                    print("epsilon:", round(self.eps, 3))
                    print("average lifetime:", avg_lifetime) 
                    print("-------------------------------")
                    
    def choose_action(self, sess, s, exploration="e-greedy"):
        
        if (exploration == "greedy"):
            a = self.agent_net.get_q_argmax(sess, [s])
        elif (exploration == "e-greedy"):
            if np.random.rand(1) < self.eps:
                a = np.random.randint(self.num_actions)
            else:
                a = self.agent_net.get_q_argmax(sess, [s])     
        elif (exploration == "boltzmann"):
            q_values = self.agent_net.get_q_values_s(sess, [s])
            logits = q_values / self.eps
            probs = softmax(logits).ravel()
            a = np.random.choice(self.num_actions, p=probs)
        elif (exploration == "policy"):
            probs = self.agent_net.get_p_values_s(sess, [s]).ravel()
            a = np.random.choice(self.num_actions, p=probs)
        else:
            return 0
        return a
                           
    def update_agent_weights(self, sess, batch):
        
        # estimate the right hand side of the Bellman equation
        agent_actions = self.agent_net.get_q_argmax(sess, batch.s_)
        q_double = self.target_net.get_q_values_sa(sess, batch.s_, agent_actions)
        targets = batch.r + (self.gamma * q_double * (1 - batch.done))

        # update agent network
        self.agent_net.update(sess, batch.s, batch.a, targets)

    def update_target_graph(self, tau):
        op_holder = []
        for agnt, trgt in zip(self.agent_vars, self.target_vars):
            op = trgt.assign(agnt.value()*tau + (1 - tau)*trgt.value())
            op_holder.append(op)
        return op_holder

    def update_target_weights(self, sess, op_holder):
        for op in op_holder:
            sess.run(op)
            
    def play(self,
             gpu_id=0,
             max_episode_length=2000,
             from_epoch=0):
        
        config = self.gpu_config(gpu_id)       
        with tf.Session(config=config) as sess:
            self.saver.restore(sess, self.path+"/model-"+str(from_epoch))
            s = self.train_env.reset()
            R = 0
            for time_step in range(max_episode_length):
                a = self.agent_net.get_q_argmax(sess, [s])[0]
                s, r, done = self.train_env.step(a)
                R += r
                self.train_env.plot_state()
                display.clear_output(wait=True)
                display.display(plt.gcf())
                if done: break
        return R
    
    def gpu_config(self, gpu_id):
        if (gpu_id == -1):
            config = tf.ConfigProto()
        else:
            os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  
            os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
            config = tf.ConfigProto()
            config.gpu_options.allow_growth = True
            config.intra_op_parallelism_threads = 1
            config.inter_op_parallelism_threads= 1
        return config
    
############################## Deep Q-Network agent ##############################

class DQNAgent(Agent):
    
    def __init__(self, env, num_actions, state_shape=[8, 8, 5], 
                 convs=[[16, 2, 1], [32, 1, 1]], 
                 fully_connected=[128],
                 activation_fn=tf.nn.relu,
                 optimizer=tf.train.AdamOptimizer(2.5e-4, epsilon=0.01/32),
                 gradient_clip=10.0,
                 save_path="rl_models", model_name="DQN"):
        
        super(DQNAgent, self).__init__(env, num_actions, 
                                       state_shape=state_shape,
                                       save_path=save_path,
                                       model_name=model_name)
        
        tf.reset_default_graph()
        self.agent_net = DeepQNetwork(self.num_actions, state_shape=state_shape,
                                      convs=convs, fully_connected=fully_connected,
                                      activation_fn=activation_fn, optimizer=optimizer, 
                                      gradient_clip=gradient_clip, scope="agent")
        self.target_net = DeepQNetwork(self.num_actions, state_shape=state_shape,
                                       convs=convs, fully_connected=fully_connected,
                                       activation_fn=activation_fn, optimizer=optimizer, 
                                       gradient_clip=gradient_clip, scope="target")
        self.init_weights()
          
########################## Dueling Deep Q-Network agent ##########################

class DuelDQNAgent(Agent):
    
    def __init__(self, env, num_actions, state_shape=[8, 8, 5], 
                 convs=[[16, 2, 1], [32, 1, 1]], 
                 fully_connected=[64],
                 activation_fn=tf.nn.relu,
                 optimizer=tf.train.AdamOptimizer(2.5e-4, epsilon=0.01/32),
                 gradient_clip=10.0,
                 save_path="rl_models", model_name="DuelDQN"):
        
        super(DuelDQNAgent, self).__init__(env, num_actions,
                                           state_shape=state_shape,
                                           save_path=save_path,
                                           model_name=model_name)
        
        tf.reset_default_graph()
        self.agent_net = DuelingDeepQNetwork(self.num_actions, state_shape=state_shape,
                                             convs=convs, fully_connected=fully_connected,
                                             activation_fn=activation_fn, optimizer=optimizer, 
                                             gradient_clip=gradient_clip, scope="agent")
        self.target_net = DuelingDeepQNetwork(self.num_actions, state_shape=state_shape,
                                              convs=convs, fully_connected=fully_connected,
                                              activation_fn=activation_fn, optimizer=optimizer, 
                                              gradient_clip=gradient_clip, scope="target")
        self.init_weights()
        
######################### Categorical Deep Q-Network agent #######################
        
class CatDQNAgent(Agent):
    
    def __init__(self, env, num_actions, state_shape=[8, 8, 5],
                 convs=[[16, 2, 1], [32, 1, 1]], 
                 fully_connected=[128],
                 activation_fn=tf.nn.relu,
                 num_atoms=21, v=(-10, 10),
                 optimizer=tf.train.AdamOptimizer(2.5e-4, epsilon=0.01/32),
                 save_path="rl_models", model_name="CatDQN"):

        super(CatDQNAgent, self).__init__(env, num_actions, 
                                          state_shape=state_shape,
                                          save_path=save_path,
                                          model_name=model_name)

        tf.reset_default_graph()
        self.agent_net = CategoricalDeepQNetwork(self.num_actions, state_shape=state_shape,
                                                 convs=convs, fully_connected=fully_connected,
                                                 activation_fn=tf.nn.relu, num_atoms=num_atoms, 
                                                 v=v, optimizer=optimizer, scope="agent")
        self.target_net = CategoricalDeepQNetwork(self.num_actions, state_shape=state_shape,
                                                  convs=convs, fully_connected=fully_connected,
                                                  activation_fn=tf.nn.relu, num_atoms=num_atoms, 
                                                  v=v, optimizer=optimizer, scope="target")
        self.init_weights()

    def update_agent_weights(self, sess, batch):

        # estimate categorical projection of the RHS of the Bellman equation
        agent_actions = self.agent_net.get_q_argmax(sess, batch.s_)
        probs_targets = self.target_net.cat_proj(sess, batch.s_, agent_actions, batch.r, 
                                                 batch.done, gamma=self.gamma)

        # update agent network
        self.agent_net.update(sess, batch.s, batch.a, probs_targets)

##################### Quantile Regression Deep Q-Network agent ###################
        
class QuantRegDQNAgent(Agent):
    
    def __init__(self, env, num_actions, state_shape=[8, 8, 5],
                 convs=[[16, 2, 1], [32, 1, 1]], 
                 fully_connected=[128],
                 activation_fn=tf.nn.relu,
                 num_atoms=50, kappa=1.0,
                 optimizer=tf.train.AdamOptimizer(2.5e-4, epsilon=0.01/32),
                 save_path="rl_models", model_name="QuantRegDQN"):

        super(QuantRegDQNAgent, self).__init__(env, num_actions,
                                               state_shape=state_shape,
                                               save_path=save_path,
                                               model_name=model_name)

        tf.reset_default_graph()
        self.agent_net = QuantileRegressionDeepQNetwork(self.num_actions, state_shape=state_shape,
                                                 convs=convs, fully_connected=fully_connected,
                                                 activation_fn=tf.nn.relu, num_atoms=num_atoms,
                                                 kappa=kappa, optimizer=optimizer, scope="agent")
        self.target_net = QuantileRegressionDeepQNetwork(self.num_actions, state_shape=state_shape,
                                                  convs=convs, fully_connected=fully_connected,
                                                  activation_fn=tf.nn.relu, num_atoms=num_atoms,
                                                  kappa=kappa, optimizer=optimizer, scope="target")
        self.init_weights()

    def update_agent_weights(self, sess, batch):

        # calculate target atoms produced by Bellman operator
        agent_actions = self.agent_net.get_q_argmax(sess, batch.s_)
        next_atoms = self.target_net.get_atoms_sa(sess, batch.s_, agent_actions)
        target_atoms = batch.r[:, None] + self.gamma * next_atoms * (1 - batch.done[:, None])

        # update agent network
        self.agent_net.update(sess, batch.s, batch.a, target_atoms)        
        
############################# Soft Actor-Critic agent ############################

class SACAgent(Agent):
    
    def __init__(self, env, num_actions, state_shape=[8, 8, 5], 
                 convs=[[16, 2, 1], [32, 1, 1]], 
                 fully_connected=[128],
                 activation_fn=tf.nn.relu,
                 temperature=1,
                 optimizers=[tf.train.AdamOptimizer(2.5e-4), 
                             tf.train.AdamOptimizer(2.5e-4),
                             tf.train.AdamOptimizer(2.5e-4)],
                 save_path="rl_models", model_name="SAC"):
        
        super(SACAgent, self).__init__(env, num_actions,
                                       state_shape=state_shape,
                                       save_path=save_path,
                                       model_name=model_name)
        
        tf.reset_default_graph()
        self.agent_net = SoftActorCriticNetwork(self.num_actions, state_shape=state_shape,
                                                convs=convs, fully_connected=fully_connected,
                                                activation_fn=activation_fn, 
                                                optimizers=optimizers, scope="agent")
        self.target_net = SoftActorCriticNetwork(self.num_actions, state_shape=state_shape,
                                                 convs=convs, fully_connected=fully_connected,
                                                 activation_fn=activation_fn, 
                                                 optimizers=optimizers, scope="target")
        self.init_weights()
        self.t = temperature
        
    def update_agent_weights(self, sess, batch):
        
        probs = self.agent_net.get_p_values_s(sess, batch.s)
        c = probs.cumsum(axis=1)
        u = np.random.rand(len(c), 1)
        actions = (u < c).argmax(axis=1)

        v_values = self.agent_net.get_v_values_s(sess, batch.s).reshape(-1)
        v_values_next = self.target_net.get_v_values_s(sess, batch.s_).reshape(-1)
        q_values = self.agent_net.get_q_values_s(sess, batch.s)
        p_logits = self.agent_net.get_p_logits_s(sess, batch.s)
        
        x = np.arange(self.batch_size)
        q_values_selected = q_values[x, actions]
        p_logits_selected = p_logits[x, actions]
        
        q_targets = batch.r / self.t + self.gamma * v_values_next * (1 - batch.done)
        v_targets = q_values_selected - p_logits_selected
        p_targets = q_values_selected - v_values

        # update agent network
        self.agent_net.update_q(sess, batch.s, batch.a, q_targets)
        self.agent_net.update_v(sess, batch.s, v_targets)
        self.agent_net.update_p(sess, batch.s, actions, p_targets)