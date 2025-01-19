from map_elites import MapElites
from ppo.args import add_ppo_args
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-name", type=str, default="map_elites_test")
    parser.add_argument("--env-name", type=str, default="Walker-v0")
    parser.add_argument("--num-iterations", type=int, default=1000)
    parser.add_argument("--population-size", type=int, default=50)
    parser.add_argument("--mutation-rate", type=float, default=0.2)
    parser.add_argument("--crossover-rate", type=float, default=0.1)
    add_ppo_args(parser)
    args = parser.parse_args()

    feature_dimensions = [(0, 25, 10), (0, 25, 10)]
    map_elites = MapElites(
        feature_dimensions=feature_dimensions,
        population_size=args.population_size,
        mutation_rate=args.mutation_rate,
        crossover_rate=args.crossover_rate,
        num_iterations=args.num_iterations,
        args=args,
        env_name=args.env_name,
    )
    grid = map_elites.run()

    print("MAP-Elites Completed!")