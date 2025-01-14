import os
import numpy as np
import json
import random
from tqdm import tqdm
from evogym import sample_robot
from stable_baselines3 import PPO
from evogym.envs import Walker 
from scipy.ndimage import label
from scipy.spatial.distance import pdist, squareform

# ----- Configuration -----
VOXEL_GRID_SIZE = (5, 5)
POPULATION_SIZE = 50
GENERATIONS = 10
MAP_RESOLUTION = 5
SIMULATION_TIME = 5
MUTATION_RATE = 0.2
CROSSOVER_RATE = 0.5
RETRY_LIMIT = 3
ENV_NAME = "Walker-v0"

# ----- Helper Functions -----
def save_json(data, save_path):
    """Save data to a JSON file."""
    with open(save_path, 'w') as f:
        json.dump(data, f, indent=4)

def pad_or_crop(robot, target_size=VOXEL_GRID_SIZE):
    """Ensure the robot matches the target voxel grid size."""
    padded_robot = np.zeros(target_size, dtype=int)
    min_x, min_y = min(robot.shape[0], target_size[0]), min(robot.shape[1], target_size[1])
    start_x, start_y = (target_size[0] - min_x) // 2, (target_size[1] - min_y) // 2
    padded_robot[start_x:start_x + min_x, start_y:start_y + min_y] = robot[:min_x, :min_y]
    return padded_robot

def is_connected(robot):
    """Check if the robot is a single connected component."""
    labeled, num_features = label(robot > 0)
    return num_features == 1

def enforce_connected(robot):
    """Remove disconnected components from the robot."""
    labeled, num_features = label(robot > 0)
    if num_features <= 1:
        return robot
    largest_label = np.argmax(np.bincount(labeled.flat)[1:]) + 1
    return np.where(labeled == largest_label, robot, 0)

def validate_robot(robot):
    """Ensure the robot is valid and connected."""
    robot = pad_or_crop(robot)
    if not is_connected(robot):
        robot = enforce_connected(robot)
    return robot

def mutate_robot(robot):
    """Apply random mutations to the robot."""
    mutated = robot.copy()
    for x in range(robot.shape[0]):
        for y in range(robot.shape[1]):
            if robot[x, y] > 0 and random.random() < MUTATION_RATE:
                mutated[x, y] = random.choice([0, 1, 2, 3, 4])
    return validate_robot(mutated)

def crossover_robots(parent_a, parent_b):
    """Perform crossover between two parent robots."""
    child = np.where(np.random.rand(*parent_a.shape) < 0.5, parent_a, parent_b)
    return validate_robot(child)

def discretize_features(active_ratio, actuator_ratio, resolution):
    """Discretize feature space into bins."""
    bin_x = min(int(active_ratio * resolution), resolution - 1)
    bin_y = min(int(actuator_ratio * resolution), resolution - 1)
    return bin_x, bin_y

def calculate_behavioral_diversity(features):
    """Calculate diversity in the behavior space."""
    return np.mean(squareform(pdist(features))) if len(features) > 1 else 0

# ----- MAP-Elites Grid -----
def create_map_grid(resolution):
    return np.full((resolution, resolution), None, dtype=object)

def update_map(map_grid, robot, resolution, fitness, features):
    bin_x, bin_y = discretize_features(features[0], features[1], resolution)
    if map_grid[bin_x, bin_y] is None or map_grid[bin_x, bin_y]['fitness'] < fitness:
        map_grid[bin_x, bin_y] = {'robot': robot, 'fitness': fitness, 'features': features}

# ----- Training -----
def train_robot(robot, save_dir):
    """Train a robot using PPO and return its fitness."""
    model_save_path = os.path.join(save_dir, "policy.zip")
    try:
        env = Walker(robot=robot)
        model = PPO("MlpPolicy", env, verbose=0)
        model.learn(total_timesteps=10000)

        fitness = env.get_total_distance()  
    except Exception as e:
        print(f"Training failed: {e}")
        fitness = -1  
    return fitness, model_save_path

# ----- Evolutionary Loop -----
def evolve():
    resolution = MAP_RESOLUTION
    map_grid = create_map_grid(resolution)
    population = [validate_robot(sample_robot(VOXEL_GRID_SIZE)[0]) for _ in range(POPULATION_SIZE)]
    
    for gen in range(GENERATIONS):
        gen_dir = f"generation_{gen + 1}"
        os.makedirs(gen_dir, exist_ok=True)
        gen_archive = []
        
        # Train and Update MAP
        for robot in tqdm(population, desc=f"Generation {gen + 1}"):
            features = (np.mean(robot > 0), np.mean(robot == 3))  # Active ratio, actuator ratio
            fitness, policy_path = train_robot(robot, gen_dir)
            update_map(map_grid, robot, resolution, fitness, features)
            gen_archive.append({'robot': robot.tolist(), 'fitness': fitness, 'features': features, 'policy_path': policy_path})
        
        # Save generation data
        save_json({'generation': gen + 1, 'archive': gen_archive}, os.path.join(gen_dir, "data.json"))
        
        # Mutation and Crossover
        elites = [cell['robot'] for cell in map_grid.flatten() if cell]
        new_population = []
        for _ in range(POPULATION_SIZE):
            if random.random() < CROSSOVER_RATE and len(elites) > 1:
                parents = random.sample(elites, 2)
                child = crossover_robots(parents[0], parents[1])
            else:
                parent = random.choice(elites)
                child = mutate_robot(parent)
            new_population.append(child)
        population = new_population

if __name__ == "__main__":
    evolve()