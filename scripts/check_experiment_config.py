from pathlib import Path
import yaml


CONFIG_PATH = Path("configs/experiments/main_experiment.yaml")


def main():
    with open(CONFIG_PATH, "r") as file:
        config = yaml.safe_load(file)

    print("EXPERIMENT CONFIGURATION")
    print("=" * 40)

    print("Experiment:", config["experiment"]["name"])
    print("Dataset:", config["dataset"]["name"])
    print("Clients:", config["dataset"]["num_clients"])
    print("Rounds:", config["training"]["num_rounds"])

    enabled_algorithms = [
        name
        for name, settings in config["algorithms"].items()
        if settings["enabled"]
    ]

    distributions = config["data_distributions"]
    seeds = config["seeds"]

    print("Algorithms:", enabled_algorithms)
    print("Distributions:", len(distributions))
    print("Seeds:", seeds)

    total_runs = (
        len(enabled_algorithms)
        * len(distributions)
        * len(seeds)
    )

    print("Total experimental runs:", total_runs)

    assert total_runs == 36
    assert config["dataset"]["num_clients"] == 20
    assert config["training"]["client_fraction"] == 1.0

    print("\nConfiguration validation passed!")


if __name__ == "__main__":
    main()