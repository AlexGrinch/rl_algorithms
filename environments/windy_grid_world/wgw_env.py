import numpy as np
import matplotlib.pyplot as plt
import time
from IPython import display


class WindyGridWorld:

    def __init__(
            self,
            grid_size=(11, 14),
            stochasticity=0.1,
            visual=False):
        self.w, self.h = grid_size
        self.stochasticity = stochasticity
        self.visual = visual

        # x position of the wall
        self.x_wall = self.w // 2
        # y position of the hole in the wall
        self.y_hole = self.h - 4

        self.reset()

    def clip_xy(self, x, y):
        """ clip coordinates if they go beyond the grid
        """
        x_ = np.clip(x, 0, self.w - 1)
        y_ = np.clip(y, 0, self.h - 1)
        return x_, y_

    def wind_shift(self, x, y):
        """ apply wind shift to areas where wind is blowing
        """
        if x == 1:
            return self.clip_xy(x, y + 1)
        elif x > 1 and x < self.x_wall:
            return self.clip_xy(x, y + 2)
        else:
            return x, y

    def move(self, a):
        """ find valid coordinates of the agent after executing action
        """
        x, y = self.pos
        self.field[x, y] = 0
        x, y = self.wind_shift(x, y)

        if a == 0:
            x_, y_ = x + 1, y
        if a == 1:
            x_, y_ = x, y + 1
        if a == 2:
            x_, y_ = x - 1, y
        if a == 3:
            x_, y_ = x, y - 1

        # check if new position does not conflict with the wall
        if x_ == self.x_wall and y != self.y_hole:
            x_, y_ = x, y
        return self.clip_xy(x_, y_)

    def get_observation(self):
        if self.visual:
            obs = np.rot90(self.field)[:, :, None]
        else:
            obs = self.pos
        return obs

    def reset(self):
        """ resets the environment
        """
        self.field = np.zeros((self.w, self.h))
        self.field[self.x_wall, :] = 1
        self.field[self.x_wall, self.y_hole] = 0
        self.field[0, 0] = 2
        self.pos = (0, 0)
        obs = self.get_observation()
        return obs

    def step(self, a):
        """ makes a step in the environment
        """

        if np.random.rand() < self.stochasticity:
            a = np.random.randint(4)

        self.field[self.pos] = 0
        self.pos = self.move(a)
        self.field[self.pos] = 2

        done = False
        reward = 0
        if self.pos == (self.w - 1, 0):
            # episode finished successfully
            done = True
            reward = 1
        next_obs = self.get_observation()
        return next_obs, reward, done

    def play_with_policy(self, policy, max_iter=100, visualize=True):
        """ play with given policy
            returns:
                episode return, number of time steps
        """
        self.reset()
        for i in range(max_iter):
            a = np.argmax(policy[self.pos])
            next_obs, reward, done = self.step(a)
            # plot grid world state
            if visualize:
                img = np.rot90(1-self.field)
                plt.imshow(img, cmap="gray")
                display.clear_output(wait=True)
                display.display(plt.gcf())
                time.sleep(0.01)
            if done:
                break
        if visualize:
            display.clear_output(wait=True)
        return reward, i+1
