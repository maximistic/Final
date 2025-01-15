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
MAP_RESOLUTION = 3
MAX_MAP_RESOLUTION = 20
GEN_RESOLUTION_INC = 5
GENERATIONS = 10
POPULATION_SIZE = 50
TOP_N_SURVIVE = 8
SIMULATION_TIME = 5

MUTATION_RATE = 0.15
CROSSOVER_RATE = 0.3

IMPROVEMENT_THRESHOLD = 1e-3  # Minimum improvement required
DIVERSITY_THRESHOLD = 0.1
FITNESS_PLATEAU_PATIENCE = 3  # Number of generations to wait before triggering early stopping
MIN_GENERATIONS = 5  # Minimum number of generations before allowing early stopping

# PPO Training Parameters
PPO_TIMESTEPS = 2048  # Number of timesteps per PPO training iteration
PPO_ITERATIONS = 5    # Number of iterations to train PPO per generation
FINE_TUNE_ITERATIONS = 2  # Number of fine-tuning iterations for best robots

class EvolutionaryOptimizer:
    def __init__(self, args):
        self.args = args
        self.save_dir = args.save_dir
        self.resolution = MAP_RESOLUTION
        self.map_grid = self.create_map_grid(self.resolution)
        self.history = []
        self.best_fitness_history = []
        self.shared_policy = None
        self.early_stopping_counter = 0
        
        # Create save directories
        os.makedirs(self.save_dir, exist_ok=True)
        self.policy_dir = os.path.join(self.save_dir, "policies")
        os.makedirs(self.policy_dir, exist_ok=True)

    def initialize_shared_policy(self, initial_robot):
        """Initialize the shared PPO policy with a basic robot"""
        policy_path = os.path.join(self.policy_dir, "shared_policy.zip")
        
        # Create and train initial policy
        self.shared_policy = PPO(
            "MlpPolicy",
            self.create_env(initial_robot),
            verbose=0,
            tensorboard_log=os.path.join(self.save_dir, "tensorboard")
        )
        
        # Initial training
        self.shared_policy.learn(total_timesteps=PPO_TIMESTEPS)
        self.shared_policy.save(policy_path)
        return policy_path

    def create_env(self, robot):
        """Create environment for a given robot"""
        return evogym.envs.Walker(robot)

    def evaluate_robot(self, robot, shared_policy):
        """Evaluate a single robot using the shared policy"""
        env = self.create_env(robot)
        obs = env.reset()
        total_reward = 0
        done = False
        
        while not done:
            action, _ = shared_policy.predict(obs, deterministic=True)
            obs, reward, done, _ = env.step(action)
            total_reward += reward
            
        return total_reward

    def batch_evaluate_robots(self, robot_list):
        """Evaluate multiple robots using the shared policy"""
        results = []
        for robot, (i, j) in tqdm(robot_list, desc="Evaluating robots"):
            fitness = self.evaluate_robot(robot, self.shared_policy)
            results.append((robot, (i, j), fitness))
        return results

    def fine_tune_policy(self, best_robots):
        """Fine-tune the shared policy on the best performing robots"""
        for robot, _, _ in best_robots[:TOP_N_SURVIVE]:
            env = self.create_env(robot)
            self.shared_policy.set_env(env)
            self.shared_policy.learn(
                total_timesteps=PPO_TIMESTEPS * FINE_TUNE_ITERATIONS,
                reset_num_timesteps=False
            )

    def should_stop_early(self):
        """Check if evolution should stop early based on fitness improvement"""
        if len(self.best_fitness_history) < MIN_GENERATIONS:
            return False
            
        recent_best = max(self.best_fitness_history[-FITNESS_PLATEAU_PATIENCE:])
        previous_best = max(self.best_fitness_history[:-FITNESS_PLATEAU_PATIENCE])
        
        relative_improvement = (recent_best - previous_best) / (abs(previous_best) + 1e-10)
        return relative_improvement < IMPROVEMENT_THRESHOLD

    def evolve(self):
        """Main evolutionary loop with improved policy sharing"""
        population = initialize_population(POPULATION_SIZE)
        
        # Initialize shared policy with first valid robot
        self.shared_policy = self.initialize_shared_policy(population[0])
        
        for gen in range(GENERATIONS):
            start_time = time()
            gen_data = {'generation': gen + 1, 'robots': [], 'metrics': {}}
            
            # Update map and evaluate population
            for robot in population:
                update_map(self.map_grid, robot, self.resolution)
                
            robot_list = [
                (self.map_grid[i, j]['robot'], (i, j)) 
                for i in range(self.resolution) 
                for j in range(self.resolution) 
                if self.map_grid[i, j]
            ]
            
            # Evaluate all robots with current shared policy
            results = self.batch_evaluate_robots(robot_list)
            
            # Sort results by fitness
            results.sort(key=lambda x: x[2], reverse=True)
            
            # Record metrics
            fitness_scores = [r[2] for r in results]
            current_best_fitness = max(fitness_scores)
            self.best_fitness_history.append(current_best_fitness)
            
            # Fine-tune shared policy on best performers
            self.fine_tune_policy(results[:TOP_N_SURVIVE])
            
            # Save current generation data
            gen_data['metrics'] = {
                'max_fitness': current_best_fitness,
                'avg_fitness': np.mean(fitness_scores),
                'generation_time': time() - start_time
            }
            
            self.history.append(gen_data)
            save_json(gen_data, os.path.join(self.save_dir, f"generation_{gen + 1}.json"))
            
            # Check for early stopping
            if self.should_stop_early():
                print(f"Early stopping triggered at generation {gen + 1}")
                break
                
            # Generate new population
            population = evolve_population(self.map_grid, self.resolution)
            
            # Save best policy
            self.shared_policy.save(os.path.join(self.policy_dir, f"policy_gen_{gen + 1}.zip"))

def create_map_grid(resolution):
    """Create an empty MAP-Elites grid."""
    return np.full((resolution, resolution), None, dtype=object)

def update_map(map_grid, robot, resolution, fitness=0):
    """Update the MAP-Elites grid with a new robot."""
    active_ratio, actuator_ratio = extract_features(robot)
    bin1, bin2 = discretize_features(active_ratio, actuator_ratio, resolution)
    
    if map_grid[bin1, bin2] is None or map_grid[bin1, bin2]['fitness'] < fitness:
        map_grid[bin1, bin2] = {
            'robot': robot,
            'fitness': fitness,
            'features': (active_ratio, actuator_ratio)
        }

def initialize_population(size):
    """Initialize a population of random robots."""
    pd = np.array([0.4, 0.1, 0.1, 0.2, 0.2])  # empty, rigid, soft, h_act, v_act
    return [sample_robot(VOXEL_GRID_SIZE, pd=pd)[0] for _ in range(size)]

def extract_features(robot):
    """Extract feature descriptors from a robot."""
    total_voxels = np.prod(robot.shape)
    active_voxels = np.sum(robot > 0)
    actuator_voxels = np.sum((robot == 3) | (robot == 4))  # Both horizontal and vertical actuators
    
    active_ratio = active_voxels / total_voxels
    actuator_ratio = actuator_voxels / active_voxels if active_voxels > 0 else 0
    return active_ratio, actuator_ratio

def discretize_features(active_ratio, actuator_ratio, resolution):
    """Convert continuous features to discrete grid coordinates."""
    bin1 = min(int(active_ratio * resolution), resolution - 1)
    bin2 = min(int(actuator_ratio * resolution), resolution - 1)
    return bin1, bin2

def evolve_population(map_grid, resolution):
    """Create a new population using mutation and crossover."""
    new_population = []
    elite_robots = [
        map_grid[i, j]['robot'] 
        for i in range(resolution) 
        for j in range(resolution) 
        if map_grid[i, j] is not None
    ]
    
    if not elite_robots:  # If no valid robots in map
        return initialize_population(POPULATION_SIZE)
    
    for _ in range(POPULATION_SIZE):
        if random.random() < CROSSOVER_RATE and len(elite_robots) > 1:
            parents = random.sample(elite_robots, 2)
            child = crossover_robots(parents[0], parents[1])
        else:
            parent = random.choice(elite_robots)
            child = mutate_robot(parent)
        
        new_population.append(child)
    
    return new_population

def mutate_robot(robot):
    """Mutate a robot's morphology."""
    mutated = robot.copy()
    for x in range(robot.shape[0]):
        for y in range(robot.shape[1]):
            if random.random() < MUTATION_RATE:
                mutated[x, y] = random.choice([0, 1, 2, 3, 4])
    
    return enforce_connected(mutated)

def crossover_robots(robot_a, robot_b):
    """Perform crossover between two parent robots."""
    mask = np.random.rand(*robot_a.shape) < 0.5
    child = np.where(mask, robot_a, robot_b)
    return enforce_connected(child)

def is_connected(robot):
    """Check if all non-empty voxels are connected."""
    if np.sum(robot > 0) == 0:
        return False
        
    # Find first non-empty voxel
    start = tuple(np.array(np.where(robot > 0))[:, 0])
    
    visited = set()
    stack = [start]
    
    while stack:
        current = stack.pop()
        if current not in visited:
            visited.add(current)
            
            # Check neighbors
            for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                nx, ny = current[0] + dx, current[1] + dy
                if (0 <= nx < robot.shape[0] and 
                    0 <= ny < robot.shape[1] and 
                    robot[nx, ny] > 0 and 
                    (nx, ny) not in visited):
                    stack.append((nx, ny))
    
    return len(visited) == np.sum(robot > 0)

def enforce_connected(robot):
    """Ensure robot is connected by removing disconnected components."""
    if is_connected(robot):
        return robot
        
    # Find largest connected component
    visited = np.zeros_like(robot, dtype=bool)
    max_size = 0
    max_component = None
    
    for i in range(robot.shape[0]):
        for j in range(robot.shape[1]):
            if robot[i, j] > 0 and not visited[i, j]:
                component = np.zeros_like(robot)
                stack = [(i, j)]
                size = 0
                
                while stack:
                    x, y = stack.pop()
                    if not visited[x, y]:
                        visited[x, y] = True
                        component[x, y] = robot[x, y]
                        size += 1
                        
                        for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                            nx, ny = x + dx, y + dy
                            if (0 <= nx < robot.shape[0] and 
                                0 <= ny < robot.shape[1] and 
                                robot[nx, ny] > 0 and 
                                not visited[nx, ny]):
                                stack.append((nx, ny))
                
                if size > max_size:
                    max_size = size
                    max_component = component
    
    return max_component if max_component is not None else robot

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

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-name", default="Walker-v0")
    parser.add_argument("--save-dir", default="evolved_walker")
    parser.add_argument("--exp-name", default="shared_policy_evolution")
    add_ppo_args(parser)
    args = parser.parse_args()
    
    optimizer = EvolutionaryOptimizer(args)
    optimizer.evolve()