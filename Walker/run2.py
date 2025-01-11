# Changes from the previous code (run1):
    # changed mutation and crossover rates to 0.3 and 0.5 from 0.1 and 0.2
    # changed the code to share ppo policy across generations and ensure the policy is reused to potentially increase fitness.
    # implemented a better mutation and crossover method to ensure diversity and better fitness
        # Context-aware mutation: adaptive mutation rate based on voxel types and based on fitness-gradient, target underperforming regions
        # feature preserving crossover: structural integrity enforced offspring (isconnected, ensureconnected )


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
GENERATIONS = 10
POPULATION_SIZE = 25
TOP_N_SURVIVE = 5
SIMULATION_TIME = 5

MUTATION_RATE = 0.3
CROSSOVER_RATE = 0.5

EARLY_STOPPING_THRESHOLD = 1e-3
DIVERSITY_THRESHOLD = 0.05
NO_IMPROVEMENT_LIMIT = 5

RETRY_LIMIT = 3

# ----- Helper Functions -----
def save_json(data, save_path):
    """Save data to JSON file with NumPy conversion."""
    def convert(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    with open(save_path, "w") as f:
        json.dump(data, f, indent=4, default=convert)

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
def mutate_robot(robot, fitness_gradient=None):
    """
    Mutate the robot morphology with context-aware and guided rules.
    
    Args:
        robot: The robot's voxel grid.
        fitness_gradient: Optional, a grid of the same shape as the robot indicating fitness contribution of each voxel.
    
    Returns:
        A mutated robot with preserved structural integrity.
    """
    mutated = robot.copy()
    for x in range(robot.shape[0]):
        for y in range(robot.shape[1]):
            if robot[x, y] == 0:
                continue
            
            voxel_type = robot[x, y]
            mutation_rate = {
                0: 0.4,  
                1: 0.2,  
                2: 0.2,  
                3: 0.1,  
                4: 0.1   
            }.get(voxel_type, 0.2)
            
            if fitness_gradient is not None:
                mutation_rate *= (1 - fitness_gradient[x, y])  
            if random.random() < mutation_rate:
                mutated[x, y] = random.choice([0, 1, 2, 3, 4]) 
    
    return pad_or_crop(mutated)

def is_connected(robot):
    """
    Check if the robot structure is connected (no disconnected components).
    """
    visited = set()
    stack = []
    for x in range(robot.shape[0]):
        for y in range(robot.shape[1]):
            if robot[x, y] > 0:
                stack.append((x, y))
                break
        if stack:
            break

    while stack:
        x, y = stack.pop()
        if (x, y) in visited or robot[x, y] == 0:
            continue
        visited.add((x, y))
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nx, ny = x + dx, y + dy
            if 0 <= nx < robot.shape[0] and 0 <= ny < robot.shape[1]:
                stack.append((nx, ny))

    return sum(robot.flatten() > 0) == len(visited)

def enforce_connected(robot):
    """
    Enforce connectivity by removing isolated components.
    """
    from scipy.ndimage import label
    
    labeled, num_features = label(robot > 0)
    largest_label = np.argmax(np.bincount(labeled.flat)[1:]) + 1
    return np.where(labeled == largest_label, robot, 0)

def crossover_robots(robot_a, robot_b):
    """
    Perform feature-preserving crossover with structural integrity constraints.
    
    Args:
        robot_a: Parent robot A.
        robot_b: Parent robot B.
    
    Returns:
        A new robot combining features of both parents.
    """
    flat_a = robot_a.flatten()
    flat_b = robot_b.flatten()
    
    crossover_points = random.sample(range(flat_a.size), random.randint(1, flat_a.size // 4))
    new_robot = flat_a.copy()
    for point in crossover_points:
        new_robot[point] = flat_b[point]

    new_robot = new_robot.reshape(robot_a.shape)
    
    if not is_connected(new_robot):
        new_robot = enforce_connected(new_robot)
    
    return pad_or_crop(new_robot)

def evolve_population(map_grid, resolution):
    """
    Create a new population using guided mutation and feature-preserving crossover.
    
    Args:
        map_grid: The MAP-Elites grid containing robot data.
        resolution: Current resolution of the grid.
    
    Returns:
        A new population of robots.
    """
    new_population = []
    elite_robots = [
        map_grid[i, j]['robot'] for i in range(resolution) for j in range(resolution) if map_grid[i, j]
    ]
    
    fitness_gradients = [
        calculate_fitness_gradient(map_grid[i, j]['robot']) for i in range(resolution) for j in range(resolution) if map_grid[i, j]
    ]
    
    for _ in range(POPULATION_SIZE):
        if random.random() < CROSSOVER_RATE and len(elite_robots) > 1:
            parents = random.sample(elite_robots, 2)
            child = crossover_robots(parents[0], parents[1])
        else:
            parent_index = random.randint(0, len(elite_robots) - 1)
            parent = elite_robots[parent_index]
            fitness_gradient = fitness_gradients[parent_index]
            child = mutate_robot(parent, fitness_gradient)
        
        new_population.append(child)
    
    return new_population

def calculate_fitness_gradient(robot):
    """
    Placeholder for calculating the fitness gradient for a robot.
    This can be based on specific fitness functions or external evaluations.
    """
    gradient = np.zeros(robot.shape)
    active_mask = robot > 0
    gradient[~active_mask] = 1.0
    return gradient


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
                    shared_policy=shared_policy  # Pass shared policy to run_ppo
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

def update_map(map_grid, robot, resolution, fitness=0):
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
    pd = np.array([0.4, 0.1, 0.1, 0.2, 0.2])        # empty, rigid ,soft, h_act, v_act. 
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
    shared_policy = None  

    for gen in range(GENERATIONS):
        start_time = time()
        gen_data = {'generation': gen + 1, 'robots': [], 'metrics': {}, 'archive': []}
        
        for robot in tqdm(population, desc=f"Generation {gen + 1}"):
            update_map(map_grid, robot, resolution)

        robot_list = [(map_grid[i, j]['robot'], (i, j)) for i in range(resolution) for j in range(resolution) if map_grid[i, j]]
        results = batch_train_robots(robot_list, args, save_dir, save_dir, shared_policy)

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

        # Find the best robot of this generation
        max_fitness_index = np.argmax(fitness_scores)
        best_robot, best_location, best_fitness = results[max_fitness_index]

        # Save the best policy and update shared_policy
        shared_policy_path = os.path.join(save_dir, "best_policy.zip")
        if best_fitness > 0:  # Save the best policy only if it has positive fitness
            best_model_save_name = f"robot_{best_location[0]}_{best_location[1]}"
            best_model_path = os.path.join(save_dir, best_model_save_name)
            best_model = PPO.load(best_model_path)
            best_model.save(shared_policy_path)  # Save the shared policy explicitly
            shared_policy = shared_policy_path

        gen_data['metrics'] = {
            'coverage': np.sum([1 for x in map_grid.flatten() if x]) / (resolution * resolution),
            'average_fitness': np.mean(fitness_scores),
            'max_fitness': best_fitness,
            'behavioral_diversity': calculate_behavioral_diversity(features),
            'structural_diversity': calculate_structural_diversity([x['morphology'] for x in gen_data['robots']]),
            'generation_time': time() - start_time
        }

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

        max_fitness = gen_data['metrics']['max_fitness']
        if max_fitness > best_fitness:
            best_fitness = max_fitness
            no_improvement_count = 0
        else:
            no_improvement_count += 1
        if no_improvement_count >= NO_IMPROVEMENT_LIMIT:
            print("No improvement detected for multiple generations. Stopping early.")
            break
        if gen_data['metrics']['coverage'] > 0.9 and resolution < MAX_MAP_RESOLUTION:
            resolution += 1
            map_grid = create_map_grid(resolution)
        population = evolve_population(map_grid, resolution)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-name", default="Walker-v0")
    parser.add_argument("--save-dir", default="draft_guide1")
    parser.add_argument("--exp-name", default="sim_cooptimization2")
    add_ppo_args(parser)
    args = parser.parse_args()
    evolve(args)