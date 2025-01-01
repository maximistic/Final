import numpy as np
from scipy.spatial.distance import pdist, squareform

def calculate_behavioral_diversity(features: list) -> float:
    """Calculate pairwise Euclidean distances in feature space."""
    return np.mean(squareform(pdist(features)))

def calculate_structural_diversity(robots: list) -> float:
    """Calculate pairwise edit distances between voxel grids."""
    distances = []
    for i, robot_a in enumerate(robots):
        for j, robot_b in enumerate(robots):
            if i < j:
                distance = np.sum(robot_a != robot_b)
                distances.append(distance)
    return np.mean(distances) if distances else 0
