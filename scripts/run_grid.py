
import argparse
import subprocess
import sys
import time
from pathlib import Path

import yaml


def load_experiment_config(config_path):
    """Load the experiment configuration."""
    with open(config_path, "r") as file:
        return yaml.safe_load(file)


def validate_config(config):
    """Validate the configuration before launching experiments."""

    required_sections = [
        "algorithms",
        "data_distributions",
        "seeds",
        "training",
        "output",
    ]

    for section in required_sections:
        if section not in config:
            raise ValueError(
                f"Missing configuration section: {section}"
            )

    enabled_algorithms = [
        name
        for name, settings in config["algorithms"].items()
        if settings.get("enabled", False)
    ]

    if not enabled_algorithms:
        raise ValueError("No algorithms are enabled.")

    if not config["data_distributions"]:
        raise ValueError("No data distributions configured.")

    if not config["seeds"]:
        raise ValueError("No random seeds configured.")

    return enabled_algorithms


def run_single_experiment(
    algorithm,
    distribution,
    seed,
    config_path,
):
    """Run one experiment using the unified experiment runner."""

    command = [
        sys.executable,
        "scripts/run_experiment.py",
        "--config",
        str(config_path),
        "--algorithm",
        algorithm,
        "--distribution",
        distribution,
        "--seed",
        str(seed),
    ]

    print("\n" + "=" * 65)
    print(f"Algorithm:    {algorithm}")
    print(f"Distribution: {distribution}")
    print(f"Seed:         {seed}")
    print("=" * 65)

    start_time = time.time()

    try:
        subprocess.run(command, check=True)

        elapsed_time = time.time() - start_time

        print(
            f"\nSUCCESS: {algorithm} | "
            f"{distribution} | seed {seed}"
        )

        return {
            "algorithm": algorithm,
            "distribution": distribution,
            "seed": seed,
            "status": "success",
            "elapsed_seconds": round(elapsed_time, 2),
            "error": "",
        }

    except subprocess.CalledProcessError as error:
        elapsed_time = time.time() - start_time

        print(
            f"\nFAILED: {algorithm} | "
            f"{distribution} | seed {seed}"
        )
        print(f"Exit code: {error.returncode}")

        return {
            "algorithm": algorithm,
            "distribution": distribution,
            "seed": seed,
            "status": "failed",
            "elapsed_seconds": round(elapsed_time, 2),
            "error": f"Process exited with code {error.returncode}",
        }


def save_grid_summary(results, config):
    """Save the grid execution summary as CSV."""

    import pandas as pd

    results_dir = (
        Path(config["output"]["results_dir"]) / "summaries"
    )
    results_dir.mkdir(parents=True, exist_ok=True)

    dataframe = pd.DataFrame(results)

    output_path = results_dir / "grid_execution_summary.csv"

    dataframe.to_csv(output_path, index=False)

    print(f"\nGrid summary saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Run the complete federated learning experiment grid."
    )

    parser.add_argument(
        "--config",
        default="configs/experiments/main_experiment.yaml",
        help="Path to the experiment configuration.",
    )

    args = parser.parse_args()

    config_path = Path(args.config)

    if not config_path.exists():
        raise FileNotFoundError(
            f"Configuration file not found: {config_path}"
        )

    config = load_experiment_config(config_path)

    enabled_algorithms = validate_config(config)

    distributions = config["data_distributions"]
    seeds = config["seeds"]

    total_experiments = (
        len(enabled_algorithms)
        * len(distributions)
        * len(seeds)
    )

    print("\nFEDERATED LEARNING EXPERIMENT GRID")
    print("=" * 65)
    print(f"Experiment name: {config['experiment']['name']}")
    print(f"Algorithms:      {enabled_algorithms}")
    print(
        f"Distributions:   "
        f"{[item['name'] for item in distributions]}"
    )
    print(f"Seeds:           {seeds}")
    print(f"Communication rounds: {config['training']['num_rounds']}")
    print(f"Total experiments: {total_experiments}")
    print("=" * 65)

    confirmation = input(
        "\nStart the complete experiment grid? (yes/no): "
    ).strip().lower()

    if confirmation not in ("yes", "y"):
        print("Experiment grid cancelled.")
        return

    results = []
    grid_start_time = time.time()

    experiment_number = 0

    for algorithm in enabled_algorithms:
        for distribution in distributions:
            distribution_name = distribution["name"]

            for seed in seeds:
                experiment_number += 1

                print(
                    f"\nExperiment "
                    f"{experiment_number}/{total_experiments}"
                )

                result = run_single_experiment(
                    algorithm=algorithm,
                    distribution=distribution_name,
                    seed=seed,
                    config_path=config_path,
                )

                results.append(result)

                # Save progress after every experiment.
                save_grid_summary(results, config)

    total_time = time.time() - grid_start_time

    successful = sum(
        result["status"] == "success"
        for result in results
    )

    failed = sum(
        result["status"] == "failed"
        for result in results
    )

    print("\n" + "=" * 65)
    print("EXPERIMENT GRID COMPLETED")
    print("=" * 65)

    print(f"Total experiments: {total_experiments}")
    print(f"Successful:        {successful}")
    print(f"Failed:            {failed}")
    print(f"Total time:        {total_time / 60:.2f} minutes")

    if failed:
        print("\nFailed experiments:")

        for result in results:
            if result["status"] == "failed":
                print(
                    f"- {result['algorithm']} | "
                    f"{result['distribution']} | "
                    f"seed {result['seed']}"
                )

    print("\nGrid execution summary saved successfully.")


if __name__ == "__main__":
    main()
