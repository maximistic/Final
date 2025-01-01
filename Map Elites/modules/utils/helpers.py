import os
import json
import numpy as np
from evogym import sample_robot

def initialize_population(size: int, voxel_grid_size=(5, 5), pd=None) -> list:
    """Initialize a population of robots."""
    pd = pd or np.array([0.2, 0.2, 0.2, 0.2, 0.2])
    return [sample_robot(voxel_grid_size, pd=pd)[0] for _ in range(size)]

def save_json(data, save_path: str) -> None:
    """Save data to JSON file."""
    with open(save_path, "w") as f:
        json.dump(data, f, indent=4)

def create_dirs(dir_list: list) -> None:
    """Create directories if they do not exist."""
    for directory in dir_list:
        os.makedirs(directory, exist_ok=True)