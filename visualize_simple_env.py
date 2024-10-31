import gym
from evogym import sample_robot
import numpy as np

# import envs from the envs folder and register them
import envs

if __name__ == '__main__':
    # Set random seed for reproducibility
    np.random.seed(42)

    # create a random robot
    body, connections = sample_robot((5,5))

    # make the SimpleWalkingEnv using gym.make and with the robot information
    env = gym.make('SimpleWalkingEnv-v0', body=body, render_mode='human')
    
    # Set metadata for render fps
    env.metadata['render_fps'] = 30

    obs, _ = env.reset()

    # step the environment for 500 iterations
    total_reward = 0
    for i in range(500):
        action = env.action_space.sample()
        ob, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        # Optional: print progress
        if i % 50 == 0:
            print(f"Step {i}, Reward: {reward}, Total Reward: {total_reward}")
        
        if terminated or truncated:
            print("Episode ended")
            obs, _ = env.reset()
            total_reward = 0
    
    env.close()