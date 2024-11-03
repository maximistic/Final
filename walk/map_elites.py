import numpy as np
import random
import json
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass

@dataclass
class RobotPerformance:
    robot: 'Robot'
    fitness: float
    behavioral_desc: Tuple[float, ...]

class Robot:
    def __init__(self, height: int, width: int, pixel_weights: Dict[str, float]):
        self.height = height
        self.width = width
        self.pixel_weights = pixel_weights
        self.morphology = None
        self.generate_random_morphology()

    def generate_random_morphology(self) -> None:
        self.morphology = np.zeros((self.height, self.width), dtype=int)
        pixel_types = list(self.pixel_weights.keys())
        weights = np.array(list(self.pixel_weights.values())) / sum(self.pixel_weights.values())

        for i in range(self.height):
            for j in range(self.width):
                pixel_type = np.random.choice(len(pixel_types), p=weights)
                self.morphology[i, j] = pixel_type

    def check_connectivity(self) -> bool:
        def flood_fill(i: int, j: int, visited: set) -> None:
            if (i, j) in visited or i < 0 or i >= self.height or j < 0 or j >= self.width:
                return
            if self.morphology[i, j] == list(self.pixel_weights.keys()).index('empty'):
                return

            visited.add((i, j))
            # Check only immediate neighbors (up, down, left, right)
            for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                flood_fill(i + di, j + dj, visited)

        start = next(((i, j) for i in range(self.height) for j in range(self.width)
                      if self.morphology[i, j] != list(self.pixel_weights.keys()).index('empty')), None)

        if not start:
            return False  # All cells are empty

        visited = set()
        flood_fill(start[0], start[1], visited)

        return all((i, j) in visited 
                   for i in range(self.height) 
                   for j in range(self.width) 
                   if self.morphology[i, j] != list(self.pixel_weights.keys()).index('empty'))

    def mutate(self, mutation_rate: float = 0.1) -> 'Robot':
        new_robot = Robot(self.height, self.width, self.pixel_weights)
        new_robot.morphology = self.morphology.copy()
        pixel_types = list(range(len(self.pixel_weights)))

        for i in range(self.height):
            for j in range(self.width):
                if random.random() < mutation_rate:
                    available_types = [t for t in pixel_types if t != new_robot.morphology[i, j]]
                    new_robot.morphology[i, j] = random.choice(available_types)

        return new_robot

    def compute_behavioral_descriptors(self) -> Tuple[float, float]:
        total_pixels = self.height * self.width
        empty_index = list(self.pixel_weights.keys()).index('empty')
        actuator_indices = [
            list(self.pixel_weights.keys()).index('horizontal_actuator'),
            list(self.pixel_weights.keys()).index('vertical_actuator')
        ]

        active_pixels = np.sum(self.morphology != empty_index)
        actuator_pixels = np.sum(np.isin(self.morphology, actuator_indices))

        active_ratio = active_pixels / total_pixels
        actuator_ratio = actuator_pixels / active_pixels if active_pixels > 0 else 0

        return active_ratio, actuator_ratio

class MAPElites:
    def __init__(self, grid_height: int, grid_width: int, pixel_weights: Dict[str, float], feature_ranges: List[Tuple[float, float]], resolution: int = 10):
        self.grid_height = grid_height
        self.grid_width = grid_width
        self.pixel_weights = pixel_weights
        self.feature_ranges = feature_ranges
        self.resolution = resolution
        self.n_features = len(feature_ranges)

        archive_shape = tuple([resolution] * self.n_features)
        self.archive = np.empty(archive_shape, dtype=object)
        self.archive_stats = {
            'total_evaluations': 0,
            'filled_cells': 0,
            'highest_fitness': float('-inf')
        }

    def get_cell_indices(self, behavioral_desc: Tuple[float, ...]) -> Tuple[int, ...]:
        indices = []
        for value, (min_val, max_val) in zip(behavioral_desc, self.feature_ranges):
            normalized = (value - min_val) / (max_val - min_val)
            idx = int(normalized * (self.resolution - 1))
            idx = max(0, min(idx, self.resolution - 1))
            indices.append(idx)
        return tuple(indices)

    def add_to_archive(self, robot: Robot, fitness: float, behavioral_desc: Tuple[float, ...]) -> bool:
        self.archive_stats['total_evaluations'] += 1
        indices = self.get_cell_indices(behavioral_desc)

        current = self.archive[indices]
        if current is None:
            self.archive[indices] = RobotPerformance(robot, fitness, behavioral_desc)
            self.archive_stats['filled_cells'] += 1
            if fitness > self.archive_stats['highest_fitness']:
                self.archive_stats['highest_fitness'] = fitness
            return True

        existing_fitness = current.fitness
        if fitness > existing_fitness:
            if self.is_diverse(behavioral_desc, current.behavioral_desc):
                self.archive[indices] = RobotPerformance(robot, fitness, behavioral_desc)
                if fitness > self.archive_stats['highest_fitness']:
                    self.archive_stats['highest_fitness'] = fitness
                return True
        return False

    def is_diverse(self, new_desc: Tuple[float, ...], existing_desc: Tuple[float, ...], threshold: float = 0.1) -> bool:
        return np.linalg.norm(np.array(new_desc) - np.array(existing_desc)) > threshold

    def get_random_elite(self) -> Optional[Robot]:
        filled_cells = [(idx, robot_perf)
                        for idx, robot_perf in np.ndenumerate(self.archive)
                        if robot_perf is not None]
        if not filled_cells:
            return None
        return random.choice(filled_cells)[1].robot

    def save_archive(self, file_path: str) -> None:
        archive_data = []
        for idx, robot_perf in np.ndenumerate(self.archive):
            if robot_perf is not None:
                archive_data.append({
                    "morphology": robot_perf.robot.morphology.tolist(),
                    "behavioral_descriptors": robot_perf.behavioral_desc,
                    "fitness": robot_perf.fitness
                })
        
        with open(file_path, 'w') as f:
            json.dump({
                "archive_stats": self.archive_stats,
                "robots": archive_data
            }, f, indent=4)

def create_initial_population(n_robots: int, height: int, width: int, pixel_weights: Dict[str, float]) -> List[Robot]:
    population = []
    while len(population) < n_robots:
        robot = Robot(height, width, pixel_weights)

        while not robot.check_connectivity():
            robot.generate_random_morphology()

        population.append(robot)

    return population

def save_robot_to_json(robot: Robot, filename: str, object_name: str = "new_object_1") -> None:
    indices = []
    types = []
    neighbors = {}

    for i in range(robot.height):
        for j in range(robot.width):
            pixel_type = robot.morphology[i, j]
            if pixel_type != list(robot.pixel_weights.keys()).index('empty'):
                indices.append(int(i * robot.width + j))
                types.append(int(pixel_type))

                neighbor_indices = []
                for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    ni, nj = i + di, j + dj
                    if 0 <= ni < robot.height and 0 <= nj < robot.width:
                        if robot.morphology[ni, nj] != list(robot.pixel_weights.keys()).index('empty'):
                            neighbor_indices.append(int(ni * robot.width + nj))

                neighbors[str(indices[-1])] = neighbor_indices

    data = {
        "grid_width": robot.width,
        "grid_height": robot.height,
        "objects": {
            object_name: {
                "indices": indices,
                "types": types,
                "neighbors": neighbors
            }
        }
    }

    with open(filename, 'w') as json_file:
        json.dump(data, json_file, indent=4)
