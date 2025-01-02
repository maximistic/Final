'''includes bin storage changes Defining Behavioral Descriptors, Discretizing the Behavioral Space, and Assigning a Robot to a Bin is clearly implemented.'''


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

# ----- Configuration -----
VOXEL_GRID_SIZE = (5, 5)
MAP_RESOLUTION = 5
MAX_MAP_RESOLUTION = 20
GEN_RESOLUTION_INC = 5
GENERATIONS = 1
POPULATION_SIZE = 50
TOP_N_SURVIVE = 5
SIMULATION_TIME = 5

# Stopping Criteria
EARLY_STOPPING_THRESHOLD = 1e-3
DIVERSITY_THRESHOLD = 0.05
NO_IMPROVEMENT_LIMIT = 5

# ----- Helper Functions -----
def save_json(data, save_path):
    """Save data to JSON file."""
    with open(save_path, "w") as f:
        json.dump(data, f, indent=4)

def calculate_behavioral_diversity(features):
    """Calculate pairwise Euclidean distances in feature space."""
    return np.mean(squareform(pdist(features)))

def calculate_structural_diversity(robots):
    """Calculate pairwise edit distances between voxel grids."""
    distances = []
    for i, robot_a in enumerate(robots):
        for j, robot_b in enumerate(robots):
            if i < j:
                distance = np.sum(robot_a != robot_b)
                distances.append(distance)
    return np.mean(distances) if distances else 0

# ----- Initialization -----
def initialize_population(size):
    """Initialize the population with random robots."""
    pd = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
    return [sample_robot(VOXEL_GRID_SIZE, pd=pd)[0] for _ in range(size)]

def extract_features(robot):
    """Calculate behavioral descriptors for a robot."""
    total_voxels = np.prod(robot.shape)
    active_voxels = np.sum(robot > 0)
    actuator_voxels = np.sum(robot == 3)
    active_ratio = active_voxels / total_voxels
    actuator_ratio = actuator_voxels / active_voxels if active_voxels > 0 else 0
    return active_ratio, actuator_ratio

# ----- Map-Elites Grid -----
def create_map_grid(resolution):
    """Create an empty grid for storing elites."""
    return np.full((resolution, resolution), None, dtype=object)

def discretize_features(active_ratio, actuator_ratio, resolution):
    """Discretize feature space into bins."""
    bin1 = min(int(active_ratio * resolution), resolution - 1)
    bin2 = min(int(actuator_ratio * resolution), resolution - 1)
    return bin1, bin2

def update_map(map_grid, robot, resolution, fitness):
    """Update the grid with a robot if it is an elite solution."""
    robot = pad_or_crop(robot)
    active_ratio, actuator_ratio = extract_features(robot)
    bin1, bin2 = discretize_features(active_ratio, actuator_ratio, resolution)
    
    # Check if the bin is empty or if the new robot has higher fitness
    if map_grid[bin1, bin2] is None or map_grid[bin1, bin2]['fitness'] < fitness:
        map_grid[bin1, bin2] = {
            'robot': robot,
            'fitness': fitness,
            'features': (active_ratio, actuator_ratio)
        }

def pad_or_crop(robot, target_size=VOXEL_GRID_SIZE):
    """Pad or crop a robot morphology to fit the voxel grid size."""
    padded_robot = np.zeros(target_size, dtype=int)
    min_x = min(robot.shape[0], target_size[0])
    min_y = min(robot.shape[1], target_size[1])
    start_x = (target_size[0] - min_x) // 2
    start_y = (target_size[1] - min_y) // 2
    padded_robot[start_x:start_x + min_x, start_y:start_y + min_y] = robot[:min_x, :min_y]
    return padded_robot

# ----- PPO Training -----
def batch_train_robots(robot_list, args, model_save_dir, structure_save_dir):
    """Train all robots in the batch using PPO."""
    results = []
    for robot, (i, j) in robot_list:
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
            print(f"Error training robot {save_name}: {e}")
        results.append((robot, (i, j), best_reward))
    return results

# ----- Evolutionary Loop -----
def evolve(args):
    """Run the evolutionary optimization process."""
    resolution = MAP_RESOLUTION
    map_grid = create_map_grid(resolution)
    population = initialize_population(POPULATION_SIZE)
    save_dir = args.save_dir
    os.makedirs(save_dir, exist_ok=True)
    history = []

    for gen in range(GENERATIONS):
        start_time = time()
        gen_data = {'generation': gen + 1, 'robots': [], 'metrics': {}, 'archive': []}
        
        for robot in tqdm(population, desc=f"Generation {gen + 1}"):
            save_name = "temp_robot"
            structure_save_path = os.path.join(save_dir, f"{save_name}.npz")
            np.savez(structure_save_path, robot, None)

            try:
                fitness = run_ppo(
                    args=args,
                    body=robot,
                    connections=None,
                    env_name=args.env_name,
                    model_save_dir=save_dir,
                    model_save_name=save_name
                )
            except Exception as e:
                fitness = -1  
                print(f"Error evaluating robot: {e}")
            update_map(map_grid, robot, resolution, fitness)

        robot_list = [(map_grid[i, j]['robot'], (i, j)) for i in range(resolution) for j in range(resolution) if map_grid[i, j]]
        results = batch_train_robots(robot_list, args, save_dir, save_dir)
        
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

        archive_data = []
        for i in range(resolution):
            for j in range(resolution):
                if map_grid[i, j] is not None:
                    archive_data.append({
                        'bin': (i, j),
                        'fitness': map_grid[i, j]['fitness'],
                        'features': map_grid[i, j]['features'],
                        'morphology': map_grid[i, j]['robot'].tolist()
                    })
        gen_data['archive'] = archive_data

        gen_data['metrics'] = {
            'coverage': np.sum([1 for x in map_grid.flatten() if x]) / (resolution * resolution),
            'average_fitness': np.mean(fitness_scores),
            'max_fitness': np.max(fitness_scores),
            'behavioral_diversity': calculate_behavioral_diversity(features),
            'structural_diversity': calculate_structural_diversity([x['morphology'] for x in gen_data['robots']]),
            'generation_time': time() - start_time
        }
        history.append(gen_data)
        save_json(gen_data, os.path.join(save_dir, f"generation_{gen + 1}.json"))

        # Adjust resolution
        if (gen + 1) % GEN_RESOLUTION_INC == 0 and resolution < MAX_MAP_RESOLUTION:
            resolution += 1
            map_grid = create_map_grid(resolution)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-name", default="Walker-v0")
    parser.add_argument("--save-dir", default="saved_data")
    parser.add_argument("--exp-name", default="sim_cooptimization")
    add_ppo_args(parser)
    args = parser.parse_args()
    evolve(args)