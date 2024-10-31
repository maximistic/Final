from envs.simple_env import SimpleWalkerEnvClass
from gym.envs.registration import register

register(
    id = 'SimpleWalkingEnv-v0',
    entry_point = 'envs.simple_env:SimpleWalkerEnvClass',
    max_episode_steps = 500
)