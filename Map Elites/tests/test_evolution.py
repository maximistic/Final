import pytest
import numpy as np
from modules.evolution.evolution import evolve
from modules.utils.helpers import initialize_population

def test_initialize_population():
    """Test population initialization."""
    size = 10
    population = initialize_population(size)
    assert len(population) == size
    assert all(isinstance(robot, np.ndarray) for robot in population)

def test_evolve_config():
    """Test the evolution process with a mock configuration."""
    mock_config = {
        "map_resolution": 5,
        "population_size": 10,
        "generations": 2,
        "save_dir": "mock_saved_data",
        "env_name": "Walker-v0",
    }
    evolve(mock_config)