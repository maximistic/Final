from ppo.run import run_ppo
import os
import numpy as np

def batch_train_robots(robot_list, config, model_save_dir, structure_save_dir):
    """Train robots in batches."""
    results = []
    for robot, (i, j) in robot_list:
        save_name = f"robot_{i}_{j}"
        np.savez(os.path.join(structure_save_dir, save_name), robot, None)
        try:
            best_reward = run_ppo(
                args=config,
                body=robot,
                connections=None,
                env_name=config["env_name"],
                model_save_dir=model_save_dir,
                model_save_name=save_name
            )
        except Exception as e:
            best_reward = -1
            print(f"Error training robot {save_name}: {e}")
        results.append((robot, (i, j), best_reward))
    return results
