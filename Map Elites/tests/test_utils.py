import pytest
import numpy as np
from modules.utils.helpers import save_json, initialize_population
from modules.utils.map_grid import extract_features, discretize_features

def test_save_json(tmpdir):
    """Test JSON saving functionality."""
    data = {"test": 123}
    file_path = tmpdir / "test.json"
    save_json(data, file_path)
    with open(file_path, "r") as f:
        loaded_data = json.load(f)
    assert loaded_data == data

def test_initialize_population():
    """Test robot population initialization."""
    population = initialize_population(5)
    assert len(population) == 5
    for robot in population:
        assert isinstance(robot, np.ndarray)

def test_extract_features():
    """Test feature extraction from robots."""
    robot = np.array([[0, 3], [1, 0]])
    active_ratio, actuator_ratio = extract_features(robot)
    assert active_ratio == 0.5
    assert actuator_ratio == 1.0

def test_discretize_features():
    """Test feature discretization."""
    active_ratio, actuator_ratio = 0.8, 0.6
    bin1, bin2 = discretize_features(active_ratio, actuator_ratio, resolution=10)
    assert bin1 == 8
    assert bin2 == 6
