import numpy as np
from .wgw_env import WindyGridWorld


class EvilWindyGridWorld(WindyGridWorld):

    def __init__(
            self,
            grid_size=(7, 10),
            stochasticity=0.1,
            visual=False):
        self.w, self.h = grid_size
        self.stochasticity = stochasticity
        self.visual = visual

        # x position of the wall
        self.x_wall = self.w // 2
        # y position of the hole in the wall
        self.y_hole = self.h - 4
        self.y_hole2 = self.h - 7

        self.reset()

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
        if x_ == self.x_wall and y != self.y_hole and y != self.y_hole2:
            x_, y_ = x, y
        return self.clip_xy(x_, y_)

    def reset(self):
        """ resets the environment
        """
        self.field = np.zeros((self.w, self.h))
        self.field[self.x_wall, :] = 1
        self.field[self.x_wall, self.y_hole] = 0
        self.field[self.x_wall, self.y_hole2] = 0
        self.field[self.x_wall + 1, self.y_hole2 + 1] = -1
        self.field[self.x_wall + 1, self.y_hole2 - 1] = -1
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
        if (self.pos == (self.x_wall + 1, self.y_hole2 + 1) or
                self.pos == (self.x_wall + 1, self.y_hole2 - 1)):
            # episode finished unsuccessfully
            done = True
            reward = -1

        next_obs = self.get_observation()
        return next_obs, reward, done
