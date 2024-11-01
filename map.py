import numpy as np
import gym
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
import scipy.ndimage as ndimage
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import logging
import torch

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class MorphologyConfig:
    grid_size: Tuple[int, int] = (5, 5)
    probs: Dict[str, float] = None
    
    def __post_init__(self):
        if self.probs is None:
            self.probs = {
                'fixed': 0.2,
                'rigid': 0.2,
                'empty': 0.2,
                'actuator': 0.4
            }

@dataclass
class EvolutionConfig:
    max_generations: int = 100
    ppo_steps: int = 5000
    archive_size: Tuple[int, int] = (10, 10)
    episode_length: int = 1000
    min_displacement: float = 0.0
    max_displacement: float = 10.0

class Archive:
    def __init__(self, shape: Tuple[int, int], min_vals: List[float], max_vals: List[float]):
        self.shape = shape
        self.min_vals = np.array(min_vals)
        self.max_vals = np.array(max_vals)
        self.fitness_map = np.full(shape, -np.inf)
        self.solutions = {}
        self.controllers = {}
        
    def add_solution(self, morphology: np.ndarray, fitness: float, behavior: List[float], 
                    controller: Optional[PPO] = None):
        indices = self._get_indices(behavior)
        if indices is not None and (self.fitness_map[indices] < fitness or np.isinf(self.fitness_map[indices])):
            self.fitness_map[indices] = fitness
            self.solutions[indices] = morphology.copy()
            if controller is not None:
                self.controllers[indices] = controller
            return True
        return False
    
    def _get_indices(self, behavior: List[float]) -> Tuple[int, int]:
        normalized = (np.array(behavior) - self.min_vals) / (self.max_vals - self.min_vals)
        if np.any(normalized < 0) or np.any(normalized > 1):
            return None
        indices = tuple((normalized * np.array(self.shape)).astype(int))
        return tuple(np.clip(indices, 0, np.array(self.shape) - 1))

class EvoGymEnvWrapper(gym.Wrapper):
    def __init__(self, env: gym.Env):
        super().__init__(env)
        self.current_morphology = None
        self.episode_steps = 0
        self.max_steps = 1000
        
    def reset(self):
        self.episode_steps = 0
        obs = self.env.reset()
        if self.current_morphology is not None:
            self.set_morphology(self.current_morphology)
        return obs
    
    def step(self, action):
        self.episode_steps += 1
        obs, reward, done, info = self.env.step(action)
        
        # Extract displacement and stability information
        displacement = obs[0]  # Assuming first observation is x-position
        instability = info.get('instability_penalty', 0)
        
        # Calculate custom reward
        reward = displacement - (0.1 * instability)
        
        # Add termination conditions
        done = done or self.episode_steps >= self.max_steps
        
        return obs, reward, done, info
    
    def set_morphology(self, morphology: np.ndarray):
        self.current_morphology = morphology.copy()
        # Implementation depends on specific EvoGym environment
        # self.env.set_morphology(morphology)

class RobotEvolution:
    def __init__(self, morph_config: MorphologyConfig, evo_config: EvolutionConfig):
        self.morph_config = morph_config
        self.evo_config = evo_config
        
        # Initialize environment
        self.env = EvoGymEnvWrapper(gym.make('YourEvoGymEnv-v0'))  # Replace with actual env
        self.env = DummyVecEnv([lambda: self.env])
        
        # Initialize archive
        self.archive = Archive(
            shape=evo_config.archive_size,
            min_vals=[evo_config.min_displacement],
            max_vals=[evo_config.max_displacement]
        )
    
    def is_fully_connected(self, morphology: np.ndarray) -> bool:
        """Check if all non-empty cells in the morphology are connected."""
        # Create binary array where 1 represents any non-empty cell
        binary = (morphology != 'empty').astype(int)
        
        # Find all connected components
        labeled, num_features = ndimage.label(binary)
        
        # Count non-empty cells
        non_empty_count = np.sum(binary)
        
        # If there's only one component and it matches the number of non-empty cells
        if num_features == 1 and np.sum(labeled > 0) == non_empty_count:
            return True
        return False
    
    def generate_morphology(self) -> np.ndarray:
        """Generate a valid, fully-connected morphology."""
        while True:
            morphology = np.random.choice(
                list(self.morph_config.probs.keys()),
                size=self.morph_config.grid_size,
                p=list(self.morph_config.probs.values())
            )
            if self.is_fully_connected(morphology):
                return morphology
    
    def train_controller(self, morphology: np.ndarray) -> Tuple[PPO, float, float]:
        """Train a PPO controller for a given morphology and evaluate its performance."""
        self.env.env_method('set_morphology', morphology)
        
        # Initialize and train PPO model
        model = PPO('MlpPolicy', self.env, verbose=0,
                   device='cuda' if torch.cuda.is_available() else 'cpu')
        model.learn(total_timesteps=self.evo_config.ppo_steps)
        
        # Evaluate the trained controller
        total_reward = 0
        final_displacement = 0
        obs = self.env.reset()
        done = False
        
        while not done:
            action, _ = model.predict(obs)
            obs, reward, done, info = self.env.step(action)
            total_reward += reward[0]
            final_displacement = obs[0][0]  # Assuming first observation is x-position
            
        return model, total_reward, final_displacement
    
    def visualize_archive(self):
        """Visualize the current state of the archive."""
        plt.figure(figsize=(10, 8))
        plt.imshow(self.archive.fitness_map.T, origin='lower', cmap='viridis',
                  extent=[self.evo_config.min_displacement, self.evo_config.max_displacement,
                         self.evo_config.min_displacement, self.evo_config.max_displacement])
        plt.colorbar(label='Fitness')
        plt.title('MAP-Elites Archive')
        plt.xlabel('Displacement')
        plt.ylabel('Height')
        plt.show()
    
    def run_evolution(self):
        """Run the main evolutionary loop."""
        for generation in range(self.evo_config.max_generations):
            logger.info(f"Generation {generation + 1}/{self.evo_config.max_generations}")
            
            # Generate new morphology
            morphology = self.generate_morphology()
            
            # Train controller and evaluate
            controller, fitness, displacement = self.train_controller(morphology)
            
            # Add to archive
            added = self.archive.add_solution(
                morphology=morphology,
                fitness=fitness,
                behavior=[displacement],
                controller=controller
            )
            
            if added:
                logger.info(f"New solution added: Fitness={fitness:.2f}, Displacement={displacement:.2f}")
            
            # Visualize progress periodically
            if (generation + 1) % 10 == 0:
                self.visualize_archive()

def main():
    # Configure evolution parameters
    morph_config = MorphologyConfig()
    evo_config = EvolutionConfig()
    
    # Initialize and run evolution
    evolution = RobotEvolution(morph_config, evo_config)
    evolution.run_evolution()

if __name__ == "__main__":
    main()