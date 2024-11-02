from gymnasium import spaces
from evogym import EvoWorld
from evogym.envs import EvoGymBase

import numpy as np
import os

class SimpleWalkerEnvClass(EvoGymBase):
    metadata = {
        'render_modes': ['human'],
        'render_fps': 30
    }
    
    def __init__(self, body, connections=None, render_mode=None):
        self.world = EvoWorld.from_json(os.path.join('Final/world_data', 'simple_walker_env.json'))
        self.world.add_from_array('robot', body, 1, 1, connections=connections)
        EvoGymBase.__init__(self, self.world)

        # Store render mode
        self.render_mode = render_mode

        num_actuators = self.get_actuator_indices('robot').size
        
        # Use reset() to get initial observation
        obs, _ = self.reset()
        obs_size = obs.size

        self.action_space = spaces.Box(low=0.6, high=1.6, shape=(num_actuators,), dtype=np.float64)
        self.observation_space = spaces.Box(low=-100.0, high=100.0, shape=(obs_size,), dtype=np.float64)
        self.default_viewer.track_objects('robot')

    def step(self, action):
        pos_1 = self.object_pos_at_time(self.get_time(), "robot")
        done = super().step({'robot': action})
        pos_2 = self.object_pos_at_time(self.get_time(), "robot")
        com_1 = np.mean(pos_1, 1)
        com_2 = np.mean(pos_2, 1)
        reward = (com_2[0] - com_1[0])

        # Always prepare observation
        obs = np.concatenate((
            self.get_vel_com_obs("robot"),
            self.get_relative_pos_obs("robot"),
        ))

        # Separate terminated and truncated
        terminated = False
        truncated = False

        if done:
            print("SIMULATION UNSTABLE... TERMINATING")
            reward -= 3.0
            terminated = True

        # Check for goal condition
        if com_2[0] > 28:
            terminated = True
            reward += 1.0

        # Return observation, reward, terminated, truncated, info
        return obs, reward, terminated, truncated, {}

    def reset(self, seed=None, options=None):
        super().reset()
        obs = np.concatenate((
            self.get_vel_com_obs("robot"),
            self.get_relative_pos_obs("robot"),
        ))
        # Return obs and info
        return obs, {}

    def render(self):
        # Customize rendering based on render_mode
        if self.render_mode == 'human':
            super().render(verbose=True)
        return None

    def close(self):
        # Ensure proper cleanup
        super().close()