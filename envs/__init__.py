from Final.envs.simple_env import SimpleWalkerEnvClass
from gymnasium.envs.registration import register

register(
    id="SimpleWalkerEnv-v0",
    entry_point="Final.envs.simple_env:SimpleWalkerEnvClass",
    max_episode_steps=500
)