import numpy as np

def extract_features(robot: np.ndarray) -> tuple:
    """Extract active and actuator voxel ratios."""
    total_voxels = np.prod(robot.shape)
    active_voxels = np.sum(robot > 0)
    actuator_voxels = np.sum(robot == 3)
    active_ratio = active_voxels / total_voxels
    actuator_ratio = actuator_voxels / active_voxels if active_voxels > 0 else 0
    return active_ratio, actuator_ratio

def discretize_features(active_ratio: float, actuator_ratio: float, resolution: int) -> tuple:
    """Discretize feature values to grid bins."""
    bin1 = min(int(active_ratio * resolution), resolution - 1)
    bin2 = min(int(actuator_ratio * resolution), resolution - 1)
    return bin1, bin2

def create_map_grid(resolution: int) -> np.ndarray:
    """Create an empty map grid."""
    return np.full((resolution, resolution), None, dtype=object)

def update_map(map_grid, robot, resolution, fitness=None):
    """Update the map grid with a robot's data."""
    active_ratio, actuator_ratio = extract_features(robot)
    bin1, bin2 = discretize_features(active_ratio, actuator_ratio, resolution)
    if map_grid[bin1, bin2] is None or (fitness is not None and map_grid[bin1, bin2]['fitness'] < fitness):
        map_grid[bin1, bin2] = {'robot': robot, 'fitness': fitness, 'features': (active_ratio, actuator_ratio)}
