from gym import spaces
from evogym import EvoWorld
from evogym.envs import EvoGymBase

import numpy as np
import os

class SimpleWalkerEnvClass(EvoGymBase):

    def __init__(self, body, connections=None):
        self.world = EvoWorld.from_json(os.path.join('world_data', 'simple_walker_env.json'))
        self.world.add_from_array('robot', body, 1, 1, connections=connections)
        EvoGymBase.__init__(self, self.world)

        num_actuators = self.get_actuator_indices('robot').size
        obs_size = self.reset().size

        self.action_space = spaces.Box(low= 0.6, high=1.6, shape=(num_actuators,), dtype=np.float)
        self.observation_space = spaces.Box(low=-100.0, high=100.0, shape=(obs_size,), dtype=np.float)
        self.default_viewer.track_objects('robot')

    def step(self, action):
        pos_1 = self.object_pos_at_time(self.get_time(), "robot")
        done = super().step({'robot': action})
        pos_2 = self.object_pos_at_time(self.get_time(), "robot")
        com_1 = np.mean(pos_1, 1)
        com_2 = np.mean(pos_2, 1)
        reward = (com_2[0] - com_1[0])
        if done:
            print("SIMULATION UNSTABLE... TERMINATING")
            reward -= 3.0
        if com_2[0] > 28:
            done = True
            reward += 1.0
            obs = np.concatenate((
            self.get_vel_com_obs("robot"),
            self.get_relative_pos_obs("robot"),
            ))
        return obs, reward, done, {}

    def reset(self):
        super().reset()
        obs = np.concatenate((
            self.get_vel_com_obs("robot"),
            self.get_relative_pos_obs("robot"),
            ))
        return obs