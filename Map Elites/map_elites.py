import numpy as np
import random
import os
import json
from ppo.run import run_ppo
from evogym import get_full_connectivity, hashable, is_connected, has_actuator

class MapElites:
    def __init__(self, feature_dimensions, population_size, mutation_rate, crossover_rate, num_iterations, args, env_name):
        self.feature_dimensions = feature_dimensions
        self.grid_shape = tuple(dim[2] for dim in feature_dimensions)
        self.grid = np.empty(self.grid_shape, dtype=object)
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.num_iterations = num_iterations
        self.args = args
        self.env_name = env_name
        self.shared_ppo_model = None  # Shared PPO policy
        self.archive_dir = "map_elites_archive"  # Directory for storing archive

        # Create archive directory
        if not os.path.exists(self.archive_dir):
            os.makedirs(self.archive_dir)

    def initialize_population(self):
        """Generate a diverse set of random solutions."""
        population = []
        for _ in range(self.population_size):
            structure = np.random.randint(0, 3, size=self.feature_dimensions[0][0])
            connectivity = get_full_connectivity(structure)
            if is_connected(structure) and has_actuator(structure):
                population.append((structure, connectivity))
        return population

    def evaluate_solution(self, structure, connectivity):
        """Evaluate a single solution using PPO."""
        fitness = run_ppo(
            self.args,
            structure,
            self.env_name,
            "map_elites_controller",
            hashable(structure),
            connections=connectivity,
            model=self.shared_ppo_model,  # Use shared PPO policy
        )
        return fitness

    def extract_features(self, structure):
        """Extract active pixel ratio and actuator ratio as features."""
        total_voxels = np.prod(structure.shape)
        active_voxels = np.sum(structure > 0)
        actuator_voxels = np.sum(structure == 3)
        active_ratio = active_voxels / total_voxels
        actuator_ratio = actuator_voxels / active_voxels if active_voxels > 0 else 0
        return [active_ratio, actuator_ratio]

    def get_feature_cell(self, feature_descriptors):
        """Map feature descriptors to grid indices."""
        indices = [
            min(max(int((fd - dim[0]) / (dim[1] - dim[0]) * (dim[2] - 1)), 0), dim[2] - 1)
            for fd, dim in zip(feature_descriptors, self.feature_dimensions)
        ]
        return tuple(indices)

    def mutate(self, structure):
        """Apply mutation to the structure."""
        mutated_structure = structure.copy()
        for _ in range(np.random.randint(1, 3)):  # Apply 1-2 mutations
            idx = np.random.randint(mutated_structure.size)
            mutated_structure.flat[idx] = np.random.randint(0, 3)  # Random material
        return mutated_structure

    def crossover(self, structure1, structure2):
        """Apply crossover to two structures."""
        mask = np.random.randint(0, 2, size=structure1.shape)
        return np.where(mask, structure1, structure2)

    def save_to_archive(self, cell, data):
        """Save robot details and policy to the archive."""
        archive_path = os.path.join(self.archive_dir, f"cell_{cell}.json")
        with open(archive_path, "w") as f:
            json.dump(data, f)

    def run(self):
        """Run the MAP-Elites algorithm."""
        # Step 1: Initialize the population
        population = self.initialize_population()

        # Initialize shared PPO policy
        self.shared_ppo_model = None

        # Step 2: Evaluate initial population and fill the grid
        for structure, connectivity in population:
            fitness = self.evaluate_solution(structure, connectivity)
            feature_descriptors = self.extract_features(structure)
            cell = self.get_feature_cell(feature_descriptors)
            if self.grid[cell] is None or fitness > self.grid[cell]["fitness"]:
                self.grid[cell] = {"structure": structure, "fitness": fitness, "features": feature_descriptors}
                self.save_to_archive(
                    cell,
                    {"structure": structure.tolist(), "fitness": fitness, "features": feature_descriptors},
                )

        # Step 3: Iterative evolution
        for iteration in range(self.num_iterations):
            print(f"Iteration {iteration + 1}/{self.num_iterations}")
            non_empty_cells = [cell for cell in np.ndindex(self.grid.shape) if self.grid[cell] is not None]
            if not non_empty_cells:
                break
            selected_cell = random.choice(non_empty_cells)
            parent = self.grid[selected_cell]["structure"]

            # Apply mutation and/or crossover
            if np.random.rand() < self.mutation_rate:
                offspring = self.mutate(parent)
            elif np.random.rand() < self.crossover_rate:
                partner_cell = random.choice(non_empty_cells)
                partner = self.grid[partner_cell]["structure"]
                offspring = self.crossover(parent, partner)
            else:
                offspring = parent.copy()

            # Evaluate offspring
            connectivity = get_full_connectivity(offspring)
            if is_connected(offspring) and has_actuator(offspring):
                fitness = self.evaluate_solution(offspring, connectivity)
                feature_descriptors = self.extract_features(offspring)
                cell = self.get_feature_cell(feature_descriptors)
                if self.grid[cell] is None or fitness > self.grid[cell]["fitness"]:
                    self.grid[cell] = {"structure": offspring, "fitness": fitness, "features": feature_descriptors}
                    self.save_to_archive(
                        cell,
                        {"structure": offspring.tolist(), "fitness": fitness, "features": feature_descriptors},
                    )

        # Return the grid
        return self.grid