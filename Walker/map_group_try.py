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
from multiprocessing import Process, Queue

# ----- Configuration -----
VOXEL_GRID_SIZE = (5, 5)
MAP_RESOLUTION = 5
GENERATIONS = 20
POPULATION_SIZE = 25
MUTATION_RATE = 0.2
CROSSOVER_RATE = 0.3
RANDOM_SAMPLING_RATE = 0.1 
FITNESS_IMPROVEMENT_THRESHOLD = 0.1 
RETRY_LIMIT = 3
MAX_VALIDATION_ATTEMPTS = 10  

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
    total_voxels = np.prod(robot.shape)
    active_voxels = np.sum(robot > 0)
    actuator_voxels = np.sum(robot == 3)
    return active_voxels / total_voxels >= 0.2 and actuator_voxels >= 1  

def mutate_robot(robot):
    """Mutate robot with validity checks and a maximum number of attempts."""
    for _ in range(MAX_VALIDATION_ATTEMPTS):
        mutated = robot.copy()
        for x in range(robot.shape[0]):
            for y in range(robot.shape[1]):
                if random.random() < MUTATION_RATE:
                    mutated[x, y] = random.randint(0, 4)
        if validate_robot(mutated):
            return mutated
    return robot 

def crossover_robots(robot_a, robot_b):
    """Perform crossover between two parent robots with validity checks and a maximum number of attempts."""
    for _ in range(MAX_VALIDATION_ATTEMPTS):
        child = np.zeros_like(robot_a)
        for x in range(robot_a.shape[0]):
            for y in range(robot_a.shape[1]):
                child[x, y] = robot_a[x, y] if random.random() < 0.5 else robot_b[x, y]
        if validate_robot(child):
            return child
    return robot_a  

def evolve_population(map_grid):
    """Evolve the population while maintaining diversity."""
    new_population = []
    elite_robots = [map_grid[i, j]['robot'] for i in range(MAP_RESOLUTION) for j in range(MAP_RESOLUTION) if map_grid[i, j]]
    
    num_random_samples = int(POPULATION_SIZE * RANDOM_SAMPLING_RATE)
    for _ in range(num_random_samples):
        new_population.append(sample_robot(VOXEL_GRID_SIZE, pd=np.array([0.3, 0.1, 0.1, 0.3, 0.2]))[0])
    
    for _ in range(POPULATION_SIZE - num_random_samples):
        if random.random() < CROSSOVER_RATE and len(elite_robots) > 1:
            parents = random.sample(elite_robots, 2)
            child = crossover_robots(parents[0], parents[1])
        else:
            parent = random.choice(elite_robots)
            child = mutate_robot(parent)
        new_population.append(child)
    return new_population

def batch_train_robots(robot_list, args, model_save_dir, structure_save_dir, shared_policy=None, failed_robots=None):
    """Train robots in batch, skipping those that consistently fail to train."""
    if failed_robots is None:
        failed_robots = set()
    results = [] 
    for robot, (i, j) in robot_list:
        if (i, j) in failed_robots:
            continue 
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
            failed_robots.add((i, j))
        results.append((robot, (i, j), best_reward))
    return results, failed_robots

def create_map_grid():
    return np.full((MAP_RESOLUTION, MAP_RESOLUTION), None, dtype=object)

def discretize_features(active_ratio, actuator_ratio):
    bin1 = min(int(active_ratio * MAP_RESOLUTION), MAP_RESOLUTION - 1)
    bin2 = min(int(actuator_ratio * MAP_RESOLUTION), MAP_RESOLUTION - 1)
    return bin1, bin2

def update_map(map_grid, robot, fitness=0):
    """Update the map grid with diversity checks."""
    robot = pad_or_crop(robot)
    active_ratio, actuator_ratio = extract_features(robot)
    bin1, bin2 = discretize_features(active_ratio, actuator_ratio)
    
    if map_grid[bin1, bin2] is None:
        map_grid[bin1, bin2] = {
            'robot': robot,
            'fitness': fitness,
            'features': (active_ratio, actuator_ratio)
        }
    else:
        current_entry = map_grid[bin1, bin2]
        if fitness > current_entry['fitness'] or (
            fitness == current_entry['fitness'] and
            calculate_behavioral_diversity([current_entry['features'], (active_ratio, actuator_ratio)]) > 0
        ):
            map_grid[bin1, bin2] = {
                'robot': robot,
                'fitness': fitness,
                'features': (active_ratio, actuator_ratio)
            }

def spread_across_feature_space(population_size, bin_size=10):
    """
    Spread robots across the feature space (active_ratio, actuator_ratio) to ensure diversity.
    
    Args:
    - population_size: The desired population size
    - bin_size: The number of bins for discretizing the feature space
    
    Returns:
    - population: A list of diverse robots
    """
    feature_bins = {i: [] for i in range(bin_size * bin_size)}
    population = []
    
    for bin_index in feature_bins.keys():
        while True:
            robot = sample_robot(VOXEL_GRID_SIZE, pd=np.array([0.2, 0.1, 0.1, 0.3, 0.3]))[0]
            active_ratio, actuator_ratio = extract_features(robot)
            active_bin = int(active_ratio * bin_size)
            actuator_bin = int(actuator_ratio * bin_size)
            active_bin = min(active_bin, bin_size - 1)
            actuator_bin = min(actuator_bin, bin_size - 1)
            bin_index = active_bin * bin_size + actuator_bin
            
            if bin_index in feature_bins:
                feature_bins[bin_index].append(robot)
                population.append(robot)
                break
    
    while len(population) < population_size:
        robot = sample_robot(VOXEL_GRID_SIZE, pd=np.array([0.2, 0.1, 0.1, 0.3, 0.3]))[0]
        population.append(robot)
    
    return population[:population_size]

def initialize_population(size, bin_size=10):
    return spread_across_feature_space(size, bin_size)

def extract_features(robot):
    total_voxels = np.prod(robot.shape)
    active_voxels = np.sum(robot > 0)
    actuator_voxels = np.sum(robot == 3)
    active_ratio = active_voxels / total_voxels
    actuator_ratio = actuator_voxels / active_voxels if active_voxels > 0 else 0
    return active_ratio, actuator_ratio

def update_shared_policy(shared_policy, best_robot, best_fitness, args, save_dir):
    """Update the shared policy only if the best robot achieves significantly higher fitness."""
    if shared_policy is None or best_fitness > shared_policy.best_fitness * (1 + FITNESS_IMPROVEMENT_THRESHOLD):
        try:
            model_save_path = os.path.join(save_dir, "shared_policy.zip")
            if shared_policy is None:
                shared_policy = PPO("MlpPolicy", args.env_name, verbose=0)
                shared_policy.best_fitness = best_fitness
            shared_policy.learn(total_timesteps=10000, reset_num_timesteps=False)
            shared_policy.save(model_save_path)
            shared_policy.best_fitness = best_fitness
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
                key = f"{i},{j}"
                archive[key] = {
                    'fitness': map_grid[i, j]['fitness'],
                    'features': map_grid[i, j]['features'],
                    'robot': map_grid[i, j]['robot'].tolist()  
                }

    robots = [entry['robot'] for entry in archive.values()]
    features = [entry['features'] for entry in archive.values()]
    data_to_save = {
        "generation": generation,
        "archive": archive,
        "behavioral_diversity": calculate_behavioral_diversity(features),
        "structural_diversity": calculate_structural_diversity(robots),
    }
    save_path = os.path.join(save_dir, f"generation_{generation:03d}.json")
    save_json(data_to_save, save_path)

def train_in_environment(env_name, robot_list, args, save_dir, queue):
    """Train robots in a specific environment and put results in the queue."""
    local_args = argparse.Namespace(**vars(args))
    local_args.env_name = env_name
    local_save_dir = os.path.join(save_dir, env_name)
    os.makedirs(local_save_dir, exist_ok=True)
    
    results, _ = batch_train_robots(robot_list, local_args, local_save_dir, local_save_dir)
    queue.put((env_name, results))

def evolve(args):
    map_grid = create_map_grid()
    population = initialize_population(POPULATION_SIZE) 
    save_dir = args.save_dir
    os.makedirs(save_dir, exist_ok=True)
    shared_policy = None

    for gen in range(GENERATIONS):
        for robot in tqdm(population, desc=f"Generation {gen + 1}"):
            update_map(map_grid, robot)

        robot_list = [(map_grid[i, j]['robot'], (i, j)) for i in range(MAP_RESOLUTION) for j in range(MAP_RESOLUTION) if map_grid[i, j]]
        
        environments = ["Walker-v0", "UpStepper-v0", "GapJumper-v0"]
        
        # Use multiprocessing to train in parallel
        processes = []
        queue = Queue()
        for env_name in environments:
            p = Process(target=train_in_environment, args=(env_name, robot_list, args, save_dir, queue))
            processes.append(p)
            p.start()
        
        for p in processes:
            p.join()
        
        # Collect results from all environments
        all_results = []
        while not queue.empty():
            all_results.append(queue.get())
        
        # Aggregate results and update shared policy
        for env_name, results in all_results:
            if results:
                best_robot_idx = np.argmax([r[2] for r in results])
                best_robot, _, best_fitness = results[best_robot_idx]
                shared_policy = update_shared_policy(shared_policy, best_robot, best_fitness, args, save_dir)

        save_generation_data(map_grid, gen + 1, save_dir)
        population = evolve_population(map_grid)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-name", default="Walker-v0")
    parser.add_argument("--save-dir", default="final_guide2")
    parser.add_argument("--exp-name", default="sim_cooptimization2")
    add_ppo_args(parser)
    args = parser.parse_args()
    evolve(args)