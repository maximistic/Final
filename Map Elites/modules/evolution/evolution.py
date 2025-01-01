import numpy as np
from time import time
from tqdm import tqdm
from modules.utils.map_grid import create_map_grid, update_map
from modules.training.ppo_training import batch_train_robots
from modules.utils.helpers import save_json
import os

def evolve(config: dict) -> None:
    """Main evolutionary loop."""
    resolution = config["map_resolution"]
    map_grid = create_map_grid(resolution)
    population = initialize_population(config["population_size"])
    history = []
    save_dir = config["save_dir"]

    for gen in range(config["generations"]):
        start_time = time()
        gen_data = {'generation': gen + 1, 'robots': [], 'metrics': {}}

        # Update grid
        for robot in tqdm(population, desc=f"Generation {gen + 1}"):
            update_map(map_grid, robot, resolution)

        # Train robots
        robot_list = [(map_grid[i, j]['robot'], (i, j)) for i in range(resolution) for j in range(resolution) if map_grid[i, j]]
        results = batch_train_robots(robot_list, config, save_dir, save_dir)

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

        # Metrics
        gen_data['metrics'] = {
            'coverage': np.sum([1 for x in map_grid.flatten() if x]) / (resolution * resolution),
            'average_fitness': np.mean(fitness_scores),
            'max_fitness': np.max(fitness_scores),
            'generation_time': time() - start_time
        }
        history.append(gen_data)
        save_json(gen_data, os.path.join(save_dir, f"generation_{gen + 1}.json"))