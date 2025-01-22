import os
import numpy as np
from stable_baselines3 import PPO
from evogym import EvoSim, EvoWorld
import evogym.envs
from ppo.eval import eval_policy


def visualize_robot(robot_path, model_path, env_name="UpStepper-v0"):
    """Visualize a single robot using its trained model."""
    if not os.path.exists(robot_path):
        print(f"Robot structure file not found: {robot_path}")
        return
    if not os.path.exists(model_path + ".zip"):
        print(f"Model file not found: {model_path}")
        return

    robot_structure = np.load(robot_path)['arr_0']

    print(f"Visualizing robot: {robot_path}")
    model = PPO.load(model_path)
    eval_policy(
        model=model,
        body=robot_structure,
        connections=None,
        env_name=env_name,
        n_evals=1,
        n_envs=1,
        render_mode="human",
    )


def main(exp_dir, env_name="UpStepper-v0"):
    """Simulate all robots in the specified directory."""
    files = [f for f in os.listdir(exp_dir) if f.endswith(".npz")]

    if not files:
        print(f"No robots found in {exp_dir}")
        return

    for robot_file in files:
        robot_path = os.path.join(exp_dir, robot_file)
        model_name = robot_file.replace(".npz", "")
        model_path = os.path.join(exp_dir, model_name)
        visualize_robot(robot_path, model_path, env_name)


if __name__ == "__main__":
    exp_dir = "draft_guide1"  # Adjust this path as needed to your experiment directory
    main(exp_dir)