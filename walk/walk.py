import numpy as np
import os
import argparse
from typing import List, Tuple, Dict
from stable_baselines3 import PPO
from Final.envs.simple_env import SimpleWalkerEnvClass
from ppo.run import run_ppo
from map_elites import MAPElites, Robot, create_initial_population, save_robot_to_json

HEIGHT = 5  
WIDTH = 5   
PIXEL_WEIGHTS = {
    'empty': 0.3,
    'rigid': 0.125,
    'soft': 0.125,
    'horizontal_actuator': 0.225,
    'vertical_actuator': 0.225
}
FEATURE_RANGES = [(0.0, 1.0), (0.0, 1.0)]  
MAP_HEIGHT = 5
MAP_WIDTH = 5

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--eval_interval', type=int, default=10000)
    parser.add_argument('--n_evals', type=int, default=5)
    parser.add_argument('--n_eval_envs', type=int, default=1)
    parser.add_argument('--verbose_ppo', type=int, default=1)
    parser.add_argument('--learning_rate', type=float, default=3e-4)
    parser.add_argument('--n_steps', type=int, default=2048)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--n_epochs', type=int, default=10)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--gae_lambda', type=float, default=0.95)
    parser.add_argument('--vf_coef', type=float, default=0.5)
    parser.add_argument('--max_grad_norm', type=float, default=0.5)
    parser.add_argument('--ent_coef', type=float, default=0.01)
    parser.add_argument('--clip_range', type=float, default=0.2)
    parser.add_argument('--total_timesteps', type=int, default=500000)
    parser.add_argument('--log_interval', type=int, default=10)
    return parser.parse_args()

def evaluate_and_train_robot(args, robot: Robot, map_elites: MAPElites) -> None:
    robot_json_path = "temp_robot.json"
    save_robot_to_json(robot, robot_json_path)

    body = robot.morphology
    connections = None  

    env_name = "SimpleWalkerEnv-v0"
    best_reward = run_ppo(
        args,
        body=body,
        env_name=env_name,
        model_save_dir="models",
        model_save_name=f"robot_{id(robot)}",
        connections=connections
    )
    behavioral_desc = robot.compute_behavioral_descriptors()
    map_elites.add_to_archive(robot, best_reward, behavioral_desc)

def main():
    args = get_args()
    map_elites = MAPElites(MAP_HEIGHT, MAP_WIDTH, PIXEL_WEIGHTS, FEATURE_RANGES)
    initial_population = create_initial_population(n_robots=10, height=HEIGHT, width=WIDTH, pixel_weights=PIXEL_WEIGHTS)
    
    for robot in initial_population:
        if robot.check_connectivity():  
            evaluate_and_train_robot(args, robot, map_elites)
    generations = 10
    for gen in range(generations):
        print(f"\n===== Generation {gen + 1} =====")
        parent_robot = map_elites.get_random_elite()
        if parent_robot is None:
            print("No elite found in archive; ending evolution.")
            break
        new_robot = parent_robot.mutate(mutation_rate=0.1)
        while not new_robot.check_connectivity():
            new_robot = parent_robot.mutate(mutation_rate=0.1)
        evaluate_and_train_robot(args, new_robot, map_elites)
    print("\n==== Final Archive Statistics ====")
    print(f"Total evaluations: {map_elites.archive_stats['total_evaluations']}")
    print(f"Filled cells in archive: {map_elites.archive_stats['filled_cells']}")
    print(f"Highest fitness achieved: {map_elites.archive_stats['highest_fitness']}")
    map_elites.save_archive("final_archive.json")

if __name__ == "__main__":
    main()
