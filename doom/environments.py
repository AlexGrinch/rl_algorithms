from vizdoom import DoomGame
from PIL import Image
import numpy as np

from IPython import display
import matplotlib.pyplot as plt

########################## Doom environment template class ##########################

class DoomEnvironment:
    
    def __init__(self, scenario):
        self.game = DoomGame()
        self.game.load_config("doom/"+scenario+".cfg")
        self.game.set_doom_scenario_path("doom/"+scenario+".wad")
        self.game.set_window_visible(False)
        self.game.init()
        self.num_actions = len(self.game.get_available_buttons())
        
    def reset(self):
        self.game.new_episode()
        game_state = self.game.get_state() 
        obs = game_state.screen_buffer
        self.h, self.w = obs.shape[1:3]
        self.current_obs = self.preprocess_obs(obs)
        self.ammo, self.health = game_state.game_variables
        return self.get_obs()
    
    def get_obs(self):
        return self.current_obs[:, :, None]
    
    def get_obs_rgb(self):
        img = self.game.get_state().screen_buffer
        img = np.rollaxis(img, 0, 3)
        img = np.reshape(img, [self.h, self.w, 3])
        return img.astype(np.uint8)
    
    def preprocess_obs(self, obs):
        img = obs
        img = np.rollaxis(img, 0, 3)
        img = np.reshape(img, [self.h, self.w, 3]).astype(np.float32)
        img = img[:, :, 0] * 0.299 + img[:, :, 1] * 0.587 + img[:, :, 2] * 0.114
        img = Image.fromarray(img)
        img = img.resize((84, 84), Image.BILINEAR)
        img = np.array(img)
        return img.astype(np.uint8)
    
    def action_to_doom(self, a):
        action = [0 for i in range(self.num_actions)]
        action[a] = 1
        return action
        
    def step(self, a):
        pass
    
    def watch_random_play(self, max_ep_length=1000, frame_skip=4):
        self.reset()
        for i in range(max_ep_length):
            a = np.random.randint(self.num_actions)
            obs, reward, done = self.step(a)
            if done: break
                
            img = self.get_obs_rgb()
            if i % frame_skip == 0:
                plt.imshow(img)
                display.clear_output(wait=True)
                display.display(plt.gcf())

####################################### Basic #######################################                         

class DoomBasic(DoomEnvironment):
    
    def __init__(self):
        super(DoomBasic, self).__init__(scenario="basic")
        
    def reset(self):
        self.game.new_episode()
        game_state = self.game.get_state() 
        obs = game_state.screen_buffer
        self.h, self.w = obs.shape[1:3]
        self.current_obs = self.preprocess_obs(obs)
        return self.get_obs()    
        
    def step(self, a):
        action = self.action_to_doom(a)
        reward = self.game.make_action(action)
        
        done = self.game.is_episode_finished()
        
        if done:
            new_obs = np.zeros_like(self.current_obs, dtype=np.uint8)
        else:
            game_state = self.game.get_state() 
            new_obs = game_state.screen_buffer
            new_obs = self.preprocess_obs(new_obs)

        self.current_obs = new_obs
        
        return self.get_obs(), reward, done                              
                          
################################## Defend the line ##################################

class DoomDefendTheLine(DoomEnvironment):
    
    def __init__(self):
        super(DoomDefendTheLine, self).__init__(scenario="defend_the_line")
        
    def step(self, a):
        action = self.action_to_doom(a)
        reward = self.game.make_action(action)
        
        done = self.game.is_episode_finished()
        
        if done:
            new_obs = np.zeros_like(self.current_obs, dtype=np.uint8)
        else:
            game_state = self.game.get_state() 
            new_obs = game_state.screen_buffer
            new_obs = self.preprocess_obs(new_obs)
            new_ammo, new_health = game_state.game_variables
            
            if (reward == 1.0): reward += 0.1
            if (new_ammo < self.ammo): reward -= 0.1
            if (new_health < self.health): reward -= 0.1
                
            self.ammo, self.health = new_ammo, new_health

        self.current_obs = new_obs
        
        return self.get_obs(), reward, done
    
################################# Defend the center #################################

class DoomDefendTheCenter(DoomEnvironment):
    
    def __init__(self):
        super(DoomDefendTheCenter, self).__init__(scenario="defend_the_center")
        
    def step(self, a):
        action = self.action_to_doom(a)
        reward = self.game.make_action(action)
        
        done = self.game.is_episode_finished()
        
        if done:
            new_obs = np.zeros_like(self.current_obs, dtype=np.uint8)
        else:
            game_state = self.game.get_state() 
            new_obs = game_state.screen_buffer
            new_obs = self.preprocess_obs(new_obs)
            new_ammo, new_health = game_state.game_variables
            
            if (reward == 1.0): reward += 0.1
            if (new_ammo < self.ammo): reward -= 0.1
            if (new_health < self.health): reward -= 0.1
                
            self.ammo, self.health = new_ammo, new_health

        self.current_obs = new_obs
        
        return self.get_obs(), reward, done