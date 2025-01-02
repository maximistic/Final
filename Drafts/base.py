# map elites + ppo - simultaneous co optimization, initial code. 
import os
import argparse
import numpy as np
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
from evogym import sample_robot, is_connected
from stable_baselines3 import PPO
from ppo.args import add_ppo_args
from ppo.run import run_ppo
import evogym.envs 

# ----- Configuration -----
VOXEL_GRID_SIZE = (5, 5)
MAP_RESOLUTION = 10
GENERATIONS = 2
POPULATION_SIZE = 50
SIMULATION_TIME = 5  

# ----- Feature Binning -----
def discretize_features(active_ratio, actuator_ratio):
    bin1 = min(int(active_ratio * MAP_RESOLUTION), MAP_RESOLUTION - 1)
    bin2 = min(int(actuator_ratio * MAP_RESOLUTION), MAP_RESOLUTION - 1)
    return bin1, bin2

# ----- Initialization -----
def initialize_population(size):
    population = []
    pd = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
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

def update_map(map_grid, robot):
    robot = pad_or_crop(robot)
    active_ratio, actuator_ratio = extract_features(robot)
    bin1, bin2 = discretize_features(active_ratio, actuator_ratio)
    if map_grid[bin1, bin2] is None or np.random.rand() < 0.2:
        print(f"Updating bin {bin1},{bin2} with new robot.")
        map_grid[bin1, bin2] = {'robot': robot}

def pad_or_crop(robot, target_size=VOXEL_GRID_SIZE):
    padded_robot = np.zeros(target_size, dtype=int)
    min_x = min(robot.shape[0], target_size[0])
    min_y = min(robot.shape[1], target_size[1])
    start_x = (target_size[0] - min_x) // 2
    start_y = (target_size[1] - min_y) // 2
    padded_robot[start_x:start_x+min_x, start_y:start_y+min_y] = robot[:min_x, :min_y]
    return padded_robot

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
        print("Mutation failed, returning original or random robot.")
        pd = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
        return robot if np.random.rand() < 0.5 else sample_robot(VOXEL_GRID_SIZE, pd=pd)[0]

# ----- PPO Training -----
def batch_train_robots(robot_list, args, model_save_dir, structure_save_dir):
    results = []
    for i, (robot, (i, j)) in enumerate(robot_list):
        save_name = f"robot_{i}_{j}"
        np.savez(os.path.join(structure_save_dir, save_name), robot, None)
        print(f"Training Robot at Bin ({i}, {j})...")

        try:
            best_reward = run_ppo(
                args=args,
                body=robot,
                connections=None,
                env_name=args.env_name,
                model_save_dir=model_save_dir,
                model_save_name=save_name
            )
            print(f"Robot {i},{j} finished with reward {best_reward}")
        except Exception as e:
            print(f"Training failed for Robot {i},{j} - Error: {e}")
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
            update_map(map_grid, robot)

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
            
            for future in futures:
                batch_results = future.result()
                for save_name, result in batch_results:
                    results[save_name] = result
        
        new_population = [mutate(map_grid[i, j]['robot']) for i in range(map_grid.shape[0]) for j in range(map_grid.shape[1]) if map_grid[i, j] is not None]

        while len(new_population) < POPULATION_SIZE:
            pd = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
            new_population.append(sample_robot(VOXEL_GRID_SIZE, pd=pd)[0])

        population = new_population

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-name", default="Walker-v0")
    parser.add_argument("--save-dir", default="saved_data")
    parser.add_argument("--exp-name", default="sim_cooptimization")
    add_ppo_args(parser)
    args = parser.parse_args()
    evolve(args)