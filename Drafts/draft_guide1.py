import os
import argparse
import numpy as np
import json
from tqdm import tqdm
from time import time
from evogym import sample_robot
from stable_baselines3 import PPO
from ppo.args import add_ppo_args
from ppo.run import run_ppo
import evogym.envs
from scipy.spatial.distance import pdist, squareform
import copy
import random

# ----- Configuration -----
VOXEL_GRID_SIZE = (5, 5)
MAP_RESOLUTION = 5
MAX_MAP_RESOLUTION = 20
GEN_RESOLUTION_INC = 5
GENERATIONS = 1
POPULATION_SIZE = 25
TOP_N_SURVIVE = 5
SIMULATION_TIME = 5

# Mutation and Crossover Parameters
MUTATION_RATE = 0.1
CROSSOVER_RATE = 0.2

# Stopping Criteria
EARLY_STOPPING_THRESHOLD = 1e-3
DIVERSITY_THRESHOLD = 0.05
NO_IMPROVEMENT_LIMIT = 5

# Retry Logic
RETRY_LIMIT = 3

# ----- Helper Functions -----
def save_json(data, save_path):
    """Save data to JSON file."""
    with open(save_path, "w") as f:
        json.dump(data, f, indent=4)

def calculate_behavioral_diversity(features):
    """Calculate pairwise Euclidean distances in feature space."""
    return np.mean(squareform(pdist(features))) if len(features) > 1 else 0

def calculate_structural_diversity(robots):
    """Calculate pairwise edit distances between voxel grids."""
    distances = []
    for i, robot_a in enumerate(robots):
        for j, robot_b in enumerate(robots):
            if i < j:
                distance = np.sum(robot_a != robot_b)
                distances.append(distance)
    return np.mean(distances) if distances else 0

def pad_or_crop(robot, target_size=VOXEL_GRID_SIZE):
    padded_robot = np.zeros(target_size, dtype=int)
    min_x = min(robot.shape[0], target_size[0])
    min_y = min(robot.shape[1], target_size[1])
    start_x = (target_size[0] - min_x) // 2
    start_y = (target_size[1] - min_y) // 2
    padded_robot[start_x:start_x + min_x, start_y:start_y + min_y] = robot[:min_x, :min_y]
    return padded_robot

# Mutation and Crossover
def mutate_robot(robot):
    """Mutate the robot morphology."""
    mutated = robot.copy()
    for x in range(robot.shape[0]):
        for y in range(robot.shape[1]):
            if random.random() < MUTATION_RATE:
                mutated[x, y] = random.randint(0, 4)  # Random voxel type
    return pad_or_crop(mutated)

def crossover_robots(robot_a, robot_b):
    """Perform crossover between two robots."""
    crossover_point = random.randint(0, robot_a.size - 1)
    flat_a = robot_a.flatten()
    flat_b = robot_b.flatten()
    new_robot = np.zeros_like(flat_a)
    new_robot[:crossover_point] = flat_a[:crossover_point]
    new_robot[crossover_point:] = flat_b[crossover_point:]
    return pad_or_crop(new_robot.reshape(robot_a.shape))

def evolve_population(map_grid, resolution):
    """Create a new population using mutation and crossover."""
    new_population = []
    elite_robots = [map_grid[i, j]['robot'] for i in range(resolution) for j in range(resolution) if map_grid[i, j]]
    for _ in range(POPULATION_SIZE):
        if random.random() < CROSSOVER_RATE and len(elite_robots) > 1:
            parents = random.sample(elite_robots, 2)
            child = crossover_robots(parents[0], parents[1])
        else:
            parent = random.choice(elite_robots)
            child = mutate_robot(parent)
        new_population.append(child)
    return new_population

# ----- PPO Training -----
def batch_train_robots(robot_list, args, model_save_dir, structure_save_dir, shared_policy=None):
    results = []
    for robot, (i, j) in robot_list:
        save_name = f"robot_{i}_{j}"
        np.savez(os.path.join(structure_save_dir, save_name), robot, None)
        retry_count = 0
        best_reward = -1
        while retry_count < RETRY_LIMIT:
            try:
                best_reward = run_ppo(
                    args=args,
                    body=robot,
                    connections=None,
                    env_name=args.env_name,
                    model_save_dir=model_save_dir,
                    model_save_name=save_name,
                    policy_init=shared_policy  # Use shared policy
                )
                break
            except Exception as e:
                retry_count += 1
                print(f"Retry {retry_count}/{RETRY_LIMIT} for robot {save_name} due to error: {e}")
        if best_reward == -1:
            print(f"Failed to train robot {save_name} after {RETRY_LIMIT} retries.")
        results.append((robot, (i, j), best_reward))
    return results

# ----- Map-Elites Grid -----
def create_map_grid(resolution):
    return np.full((resolution, resolution), None, dtype=object)

def discretize_features(active_ratio, actuator_ratio, resolution):
    bin1 = min(int(active_ratio * resolution), resolution - 1)
    bin2 = min(int(actuator_ratio * resolution), resolution - 1)
    return bin1, bin2

def update_map(map_grid, robot, resolution, fitness):
    robot = pad_or_crop(robot)
    active_ratio, actuator_ratio = extract_features(robot)
    bin1, bin2 = discretize_features(active_ratio, actuator_ratio, resolution)
    
    if map_grid[bin1, bin2] is None or map_grid[bin1, bin2]['fitness'] < fitness:
        map_grid[bin1, bin2] = {
            'robot': robot,
            'fitness': fitness,
            'features': (active_ratio, actuator_ratio)
        }

# ----- Initialization -----
def initialize_population(size):
    pd = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
    return [sample_robot(VOXEL_GRID_SIZE, pd=pd)[0] for _ in range(size)]

def extract_features(robot):
    total_voxels = np.prod(robot.shape)
    active_voxels = np.sum(robot > 0)
    actuator_voxels = np.sum(robot == 3)
    active_ratio = active_voxels / total_voxels
    actuator_ratio = actuator_voxels / active_voxels if active_voxels > 0 else 0
    return active_ratio, actuator_ratio

# ----- Evolutionary Loop -----
def evolve(args):
    resolution = MAP_RESOLUTION
    map_grid = create_map_grid(resolution)
    population = initialize_population(POPULATION_SIZE)
    save_dir = args.save_dir
    os.makedirs(save_dir, exist_ok=True)
    history = []
    no_improvement_count = 0
    best_fitness = -float('inf')

    for gen in range(GENERATIONS):
        start_time = time()
        gen_data = {'generation': gen + 1, 'robots': [], 'metrics': {}, 'archive': []}
        
        # Update grid
        for robot in tqdm(population, desc=f"Generation {gen + 1}"):
            update_map(map_grid, robot, resolution)

        # Train robots
        robot_list = [(map_grid[i, j]['robot'], (i, j)) for i in range(resolution) for j in range(resolution) if map_grid[i, j]]
        shared_policy = None if gen == 0 else shared_policy  # Use the best policy from the previous generation
        results = batch_train_robots(robot_list, args, save_dir, save_dir, shared_policy)

        # Collect data
        fitness_scores = []
        features = []
        for robot, (i, j), reward in results:
            fitness_scores.append(reward)
            features.append(map_grid[i, j]['features'])
            gen_data['robots'].append({
                'bin': (i, j),
                'morphology': robot.tolist(),
                'fitness': reward,
                'features': map_grid[i, j]['features']
            })

        # Calculate Metrics
        gen_data['metrics'] = {
            'coverage': np.sum([1 for x in map_grid.flatten() if x]) / (resolution * resolution),
            'average_fitness': np.mean(fitness_scores),
            'max_fitness': np.max(fitness_scores),
            'behavioral_diversity': calculate_behavioral_diversity(features),
            'structural_diversity': calculate_structural_diversity([x['morphology'] for x in gen_data['robots']]),
            'generation_time': time() - start_time
        }

        # Archive the map
        gen_data['archive'] = [
            {
                'bin': (i, j),
                'robot': cell['robot'].tolist(),
                'fitness': cell['fitness'],
                'features': cell['features']
            } for i in range(resolution) for j in range(resolution) if (cell := map_grid[i, j])
        ]

        history.append(gen_data)
        save_json(gen_data, os.path.join(save_dir, f"generation_{gen + 1}.json"))

        # Check for improvement
        max_fitness = gen_data['metrics']['max_fitness']
        if max_fitness > best_fitness:
            best_fitness = max_fitness
            no_improvement_count = 0
        else:
            no_improvement_count += 1

        # Early stopping
        if no_improvement_count >= NO_IMPROVEMENT_LIMIT:
            print("No improvement detected for multiple generations. Stopping early.")
            break

        # Adjust resolution dynamically
        if gen_data['metrics']['coverage'] > 0.9 and resolution < MAX_MAP_RESOLUTION:
            resolution += 1
            map_grid = create_map_grid(resolution)

        # Generate new population
        population = evolve_population(map_grid, resolution)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-name", default="Walker-v0")
    parser.add_argument("--save-dir", default="draft_guide1")
    parser.add_argument("--exp-name", default="sim_cooptimization2")
    add_ppo_args(parser)
    args = parser.parse_args()
    evolve(args)