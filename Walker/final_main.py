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
import random

# ----- Configuration -----
VOXEL_GRID_SIZE = (5, 5)
MAP_RESOLUTION = 10  
GENERATIONS = 10  
POPULATION_SIZE = 25
MUTATION_RATE = 0.2
CROSSOVER_RATE = 0.3

RETRY_LIMIT = 3

# ----- Helper Functions -----
def save_json(data, save_path):
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
    return np.mean(squareform(pdist(features))) if len(features) > 1 else 0

def calculate_structural_diversity(robots):
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

def validate_robot(robot):
    """Ensure robot is valid (enough active voxels, contains actuators)."""
    total_voxels = np.prod(robot.shape)
    active_voxels = np.sum(robot > 0)
    actuator_voxels = np.sum(robot == 3)
    return active_voxels / total_voxels >= 0.2 and actuator_voxels > 0

def mutate_robot(robot):
    """Mutate robot with validity checks."""
    while True:
        mutated = robot.copy()
        for x in range(robot.shape[0]):
            for y in range(robot.shape[1]):
                if random.random() < MUTATION_RATE:
                    mutated[x, y] = random.randint(0, 4)  
        mutated = pad_or_crop(mutated)
        if validate_robot(mutated):
            return mutated

def crossover_robots(robot_a, robot_b):
    """Multi-point crossover with validity checks."""
    while True:
        flat_a = robot_a.flatten()
        flat_b = robot_b.flatten()
        new_robot = np.zeros_like(flat_a)

        # Multi-point crossover
        points = sorted(random.sample(range(robot_a.size), k=3))
        start = 0
        for i, point in enumerate(points + [robot_a.size]):
            if i % 2 == 0:
                new_robot[start:point] = flat_a[start:point]
            else:
                new_robot[start:point] = flat_b[start:point]
            start = point

        new_robot = pad_or_crop(new_robot.reshape(robot_a.shape))
        if validate_robot(new_robot):
            return new_robot

def evolve_population(map_grid):
    new_population = []
    elite_robots = [map_grid[i, j]['robot'] for i in range(MAP_RESOLUTION) for j in range(MAP_RESOLUTION) if map_grid[i, j]]
    for _ in range(POPULATION_SIZE):
        if random.random() < CROSSOVER_RATE and len(elite_robots) > 1:
            parents = random.sample(elite_robots, 2)
            child = crossover_robots(parents[0], parents[1])
        else:
            parent = random.choice(elite_robots)
            child = mutate_robot(parent)
        new_population.append(child)
    return new_population

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
                    shared_policy=shared_policy
                )
                break
            except Exception as e:
                retry_count += 1
                print(f"Retry {retry_count}/{RETRY_LIMIT} for robot {save_name} due to error: {e}")
        if best_reward == -1:
            print(f"Failed to train robot {save_name} after {RETRY_LIMIT} retries.")
        results.append((robot, (i, j), best_reward))
    return results

def create_map_grid():
    return np.full((MAP_RESOLUTION, MAP_RESOLUTION), None, dtype=object)

def discretize_features(active_ratio, actuator_ratio):
    bin1 = min(int(active_ratio * MAP_RESOLUTION), MAP_RESOLUTION - 1)
    bin2 = min(int(actuator_ratio * MAP_RESOLUTION), MAP_RESOLUTION - 1)
    return bin1, bin2

def update_map(map_grid, robot, fitness=0):
    robot = pad_or_crop(robot)
    active_ratio, actuator_ratio = extract_features(robot)
    bin1, bin2 = discretize_features(active_ratio, actuator_ratio)
    if map_grid[bin1, bin2] is None or map_grid[bin1, bin2]['fitness'] < fitness:
        map_grid[bin1, bin2] = {
            'robot': robot,
            'fitness': fitness,
            'features': (active_ratio, actuator_ratio)
        }

def initialize_population(size):
    pd = np.array([0.4, 0.1, 0.1, 0.2, 0.2])
    return [sample_robot(VOXEL_GRID_SIZE, pd=pd)[0] for _ in range(size)]

def extract_features(robot):
    total_voxels = np.prod(robot.shape)
    active_voxels = np.sum(robot > 0)
    actuator_voxels = np.sum(robot == 3)
    active_ratio = active_voxels / total_voxels
    actuator_ratio = actuator_voxels / active_voxels if active_voxels > 0 else 0
    return active_ratio, actuator_ratio

def update_shared_policy(shared_policy, best_robot, args, save_dir):
    """Incrementally update shared policy with fine-tuning."""
    try:
        model_save_path = os.path.join(save_dir, "shared_policy.zip")
        if shared_policy is None:
            shared_policy = PPO("MlpPolicy", args.env_name, verbose=0)
        shared_policy.learn(total_timesteps=10000, reset_num_timesteps=False)
        shared_policy.save(model_save_path)
        return PPO.load(model_save_path)
    except Exception as e:
        print(f"Failed to update shared policy: {e}")
        return shared_policy

def save_generation_data(map_grid, generation, save_dir):
    """Save archive, robot morphologies, fitness, and metrics for the current generation."""
    archive = {}
    for i in range(MAP_RESOLUTION):
        for j in range(MAP_RESOLUTION):
            if map_grid[i, j] is not None:
                archive[(i, j)] = {
                    'fitness': map_grid[i, j]['fitness'],
                    'features': map_grid[i, j]['features'],
                    'robot': map_grid[i, j]['robot'].tolist()  # Convert numpy array to list for JSON
                }

    # Add additional metrics
    robots = [entry['robot'] for entry in archive.values()]
    features = [entry['features'] for entry in archive.values()]
    data_to_save = {
        "generation": generation,
        "archive": archive,
        "behavioral_diversity": calculate_behavioral_diversity(features),
        "structural_diversity": calculate_structural_diversity(robots),
    }

    # Save the data to a JSON file
    save_path = os.path.join(save_dir, f"generation_{generation:03d}.json")
    save_json(data_to_save, save_path)

def evolve(args):
    map_grid = create_map_grid()
    population = initialize_population(POPULATION_SIZE)
    save_dir = args.save_dir
    os.makedirs(save_dir, exist_ok=True)
    shared_policy = None

    for gen in range(GENERATIONS):
        start_time = time()
        for robot in tqdm(population, desc=f"Generation {gen + 1}"):
            update_map(map_grid, robot)

        robot_list = [(map_grid[i, j]['robot'], (i, j)) for i in range(MAP_RESOLUTION) for j in range(MAP_RESOLUTION) if map_grid[i, j]]
        results = batch_train_robots(robot_list, args, save_dir, save_dir, shared_policy)

        if results:
            best_robot_idx = np.argmax([r[2] for r in results])
            best_robot, _, _ = results[best_robot_idx]
            shared_policy = update_shared_policy(shared_policy, best_robot, args, save_dir)

        # Save JSON data for the current generation
        save_generation_data(map_grid, gen + 1, save_dir)

        # Evolve the population for the next generation
        population = evolve_population(map_grid)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-name", default="Walker-v0")
    parser.add_argument("--save-dir", default="final_guide")
    parser.add_argument("--exp-name", default="sim_cooptimization2")
    add_ppo_args(parser)
    args = parser.parse_args()
    evolve(args)