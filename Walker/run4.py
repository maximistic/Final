# no json file


import os
import shutil
import json
import argparse
import numpy as np
import pickle
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
from evogym import EvoWorld, EvoSim, sample_robot, is_connected
from stable_baselines3 import PPO
from ppo.args import add_ppo_args
from ppo.run import run_ppo
import evogym.envs 

# ----- Configuration -----
VOXEL_GRID_SIZE = (5, 5)
MAP_RESOLUTION = 10
GENERATIONS = 10
POPULATION_SIZE = 25

# ----- Feature Binning -----
def discretize_features(active_ratio, actuator_ratio):
    bin1 = min(int(active_ratio * MAP_RESOLUTION), MAP_RESOLUTION - 1)
    bin2 = min(int(actuator_ratio * MAP_RESOLUTION), MAP_RESOLUTION - 1)
    return bin1, bin2

# ----- Initialization -----
def initialize_population(size):
    population = []
    pd = np.array([0.4, 0.1, 0.1, 0.2, 0.2])
    for _ in range(size):
        robot, _ = sample_robot(VOXEL_GRID_SIZE, pd=pd)
        population.append(robot)
    return population

def extract_features(robot):
    total_voxels = np.prod(robot.shape)
    active_voxels = np.sum(robot > 0)
    actuator_voxels = np.sum(robot == 3)
    active_ratio = active_voxels / total_voxels
    actuator_ratio = actuator_voxels / active_voxels if active_voxels > 0 else 0
    return active_ratio, actuator_ratio

# ----- Map-Elites Grid -----
def create_map_grid():
    return np.full((MAP_RESOLUTION, MAP_RESOLUTION), None, dtype=object)

def update_map(map_grid, robot, fitness):
    robot = pad_or_crop(robot)
    active_ratio, actuator_ratio = extract_features(robot)
    bin1, bin2 = discretize_features(active_ratio, actuator_ratio)
    
    if map_grid[bin1, bin2] is None or map_grid[bin1, bin2]['fitness'] < fitness:
        map_grid[bin1, bin2] = {'robot': robot, 'fitness': fitness}

def pad_or_crop(robot, target_size=VOXEL_GRID_SIZE):
    padded_robot = np.zeros(target_size, dtype=int)
    min_x = min(robot.shape[0], target_size[0])
    min_y = min(robot.shape[1], target_size[1])
    start_x = (target_size[0] - min_x) // 2
    start_y = (target_size[1] - min_y) // 2
    padded_robot[start_x:start_x+min_x, start_y:start_y+min_y] = robot[:min_x, :min_y]
    return padded_robot

# ----- Crossover and Mutation -----
def crossover(parent1, parent2):
    crossover_point = np.random.randint(0, VOXEL_GRID_SIZE[0])
    child = np.copy(parent1)
    child[crossover_point:] = parent2[crossover_point:]
    
    if is_connected(child):
        return child
    else:
        return crossover(parent1, parent2) 

def mutate(robot):
    mutated_robot = np.copy(robot)
    mutation_cache = []
    mutation_rate = np.random.randint(1, 4)

    for _ in range(mutation_rate):
        x, y = np.random.randint(0, VOXEL_GRID_SIZE[0]), np.random.randint(0, VOXEL_GRID_SIZE[1])
        mutated_robot[x, y] = np.random.choice([0, 1, 2, 3])
        if is_connected(mutated_robot):
            mutation_cache.append(np.copy(mutated_robot))
    
    if mutation_cache:
        return mutation_cache[np.random.randint(len(mutation_cache))]
    else:
        return mutate(robot) 

# ----- PPO Training -----
def batch_train_robots(robot_list, args, model_save_dir, structure_save_dir):
    results = []
    for i, (robot, (i, j)) in enumerate(robot_list):
        save_name = f"robot_{i}_{j}"
        np.savez(os.path.join(structure_save_dir, save_name), robot, None)

        try:
            best_reward = run_ppo(
                args=args,
                body=robot,
                connections=None,
                env_name=args.env_name,
                model_save_dir=model_save_dir,
                model_save_name=save_name
            )
        except Exception as e:
            best_reward = -1
        
        results.append((save_name, {"best_reward": best_reward, "bin": (i, j)}))
    return results

# ----- Evolutionary Loop -----
def evolve(args):
    map_grid = create_map_grid()
    population = initialize_population(POPULATION_SIZE)

    exp_dir = os.path.join(args.save_dir, args.exp_name)
    model_save_dir = os.path.join(exp_dir, "controllers")
    structure_save_dir = os.path.join(exp_dir, "structures")
    os.makedirs(model_save_dir, exist_ok=True)
    os.makedirs(structure_save_dir, exist_ok=True)

    results = {}

    for gen in range(GENERATIONS):
        print(f"\nGeneration {gen + 1}/{GENERATIONS}")
        for robot in tqdm(population):
            update_map(map_grid, robot, fitness=0) 

        max_workers = min(8, os.cpu_count())
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = []
            robot_list = []

            for i in range(map_grid.shape[0]):
                for j in range(map_grid.shape[1]):
                    if map_grid[i, j] is not None:
                        robot_list.append((map_grid[i, j]['robot'], (i, j)))

            batch_size = 10
            for k in range(0, len(robot_list), batch_size):
                futures.append(
                    executor.submit(
                        batch_train_robots,
                        robot_list[k:k + batch_size],
                        args,
                        model_save_dir,
                        structure_save_dir
                    )
                )
            
            for future in as_completed(futures):
                batch_results = future.result()
                for save_name, result in batch_results:
                    results[save_name] = result

            for i in range(map_grid.shape[0]):
                for j in range(map_grid.shape[1]):
                    if map_grid[i, j] is not None:
                        robot = map_grid[i, j]['robot']
                        fitness = results.get(f"robot_{i}_{j}", {}).get('best_reward', 0)
                        update_map(map_grid, robot, fitness)

        sorted_robots = sorted([(key, value) for key, value in results.items()], key=lambda x: x[1]['best_reward'], reverse=True)
        selected_robots = [robot for robot, _ in sorted_robots[:POPULATION_SIZE]]

        if len(selected_robots) % 2 != 0:
            selected_robots.append(selected_robots[-1])  
        
        new_population = []
        for i in range(0, POPULATION_SIZE, 2):
            parent1, parent2 = selected_robots[i], selected_robots[i+1]
            child = crossover(parent1, parent2)
            new_population.append(child)
        new_population = [mutate(robot) for robot in new_population]

        population = new_population

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-name", default="Walker-v0")
    parser.add_argument("--save-dir", default="saved_data")
    parser.add_argument("--exp-name", default="sim_cooptimization")
    add_ppo_args(parser)
    args = parser.parse_args()
    evolve(args)