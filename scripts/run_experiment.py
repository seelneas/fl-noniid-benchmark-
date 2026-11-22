
import argparse
from pathlib import Path

import yaml
import copy
import torch
from flbench.utils import set_seed, get_device
from flbench.data.datasets import prepare_federated_data
from flbench.models.cnn import SimpleCNN
from flbench.fl.server import FederatedServer
import pandas as pd

from flbench.fl.simulation import run_federated_simulation
from flbench.algorithms.fedprox_simulation import run_fedprox_simulation
from flbench.algorithms.scaffold_simulation import run_scaffold_simulation


def load_experiment_config(config_path):
    """Load the experiment configuration from a YAML file."""
    with open(config_path, "r") as file:
        return yaml.safe_load(file)


def validate_algorithm(algorithm, config):
    """Check that the requested algorithm is enabled."""
    algorithms = config["algorithms"]

    if algorithm not in algorithms:
        raise ValueError(f"Unknown algorithm: {algorithm}")

    if not algorithms[algorithm].get("enabled", False):
        raise ValueError(f"{algorithm} is disabled in the configuration.")


def prepare_experiment(config, distribution, seed):
    """Prepare data, model, server, and training settings for one run."""

    set_seed(seed)
    device = get_device()

    distribution_config = next(
        (
            item for item in config["data_distributions"]
            if item["name"] == distribution
        ),
        None,
    )

    if distribution_config is None:
        raise ValueError(f"Unknown data distribution: {distribution}")

    client_loaders, test_loader, client_indices, validation_loader = (
        prepare_federated_data(
            num_clients=config["dataset"]["num_clients"],
            partition=distribution_config["partition"],
            alpha=distribution_config.get("alpha", 0.5),
            batch_size=config["dataset"]["batch_size"],
            train_val_split=config["dataset"]["train_val_split"],
            seed=seed,
        )
    )
    model = SimpleCNN(
        num_classes=config["model"]["num_classes"]
    )

    server = FederatedServer(
        global_model=model,
        device=device,
    )

    training = config["training"]
    optimizer = config["optimizer"]

    return {
        "server": server,
        "client_loaders": client_loaders,
        "validation_loader": validation_loader,
        "test_loader": test_loader,
        "client_indices": client_indices,
        "device": device,
        "training": training,
        "optimizer": optimizer,
    }


def run_selected_algorithm(algorithm, experiment, config):
    """Run the selected federated learning algorithm."""

    server = experiment["server"]
    training = experiment["training"]
    optimizer = experiment["optimizer"]

    common_arguments = {
        "server": server,
        "client_loaders": experiment["client_loaders"],
        "validation_loader": experiment["validation_loader"],
        "num_rounds": training["num_rounds"],
        "local_epochs": training["local_epochs"],
        "learning_rate": optimizer["learning_rate"],
        "momentum": optimizer["momentum"],
        "weight_decay": optimizer["weight_decay"],
    }

    if algorithm == "fedavg":
        result = run_federated_simulation(**common_arguments)

    elif algorithm == "fedprox":
        result = run_fedprox_simulation(
            **common_arguments,
            mu=config["algorithms"]["fedprox"]["mu"],
        )

    elif algorithm == "scaffold":
        result = run_scaffold_simulation(**common_arguments)

    else:
        raise ValueError(f"Unsupported algorithm: {algorithm}")

    return result


def save_experiment_results(
    history,
    algorithm,
    distribution,
    seed,
    config,
):
    """Save round-by-round experiment metrics to CSV."""

    results_dir = Path(config["output"]["results_dir"]) / "summaries"
    results_dir.mkdir(parents=True, exist_ok=True)

    dataframe = pd.DataFrame(history)

    dataframe["algorithm"] = algorithm
    dataframe["distribution"] = distribution
    dataframe["seed"] = seed

    filename = f"{algorithm}_{distribution}_seed_{seed}.csv"
    output_path = results_dir / filename

    dataframe.to_csv(output_path, index=False)

    print(f"\nResults saved to: {output_path}")


def save_test_results(
    test_metrics,
    algorithm,
    distribution,
    seed,
    result,
    config,
):
    """Save final test-set metrics for the best validation model."""

    results_dir = Path(config["output"]["results_dir"]) / "summaries"
    results_dir.mkdir(parents=True, exist_ok=True)

    test_results = {
        "algorithm": algorithm,
        "distribution": distribution,
        "seed": seed,
        "best_round": result["best_round"],
        "best_validation_accuracy": result["best_validation_accuracy"],
        "test_loss": test_metrics["loss"],
        "test_accuracy": test_metrics["accuracy"],
    }

    dataframe = pd.DataFrame([test_results])

    filename = f"{algorithm}_{distribution}_seed_{seed}_test.csv"
    output_path = results_dir / filename

    dataframe.to_csv(output_path, index=False)

    print(f"Final test results saved to: {output_path}")


def save_best_checkpoint(result, algorithm, distribution, seed):
    """Save the model from the round with the best validation accuracy."""

    checkpoint_dir = Path("checkpoints")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_path = checkpoint_dir / (
        f"{algorithm}_{distribution}_seed_{seed}.pt"
    )

    torch.save(
        {
            "model_state_dict": result["best_model_state"],
            "best_round": result["best_round"],
            "best_validation_accuracy": result["best_validation_accuracy"],
            "algorithm": algorithm,
            "distribution": distribution,
            "seed": seed,
        },
        checkpoint_path,
    )

    print(f"\nBest validation round: {result['best_round']}")
    print(
        f"Best validation accuracy: "
        f"{result['best_validation_accuracy']:.4f}"
    )
    print(f"Checkpoint saved to: {checkpoint_path}")



def main():
    parser = argparse.ArgumentParser(
        description="Run one federated learning experiment."
    )

    parser.add_argument(
        "--config",
        default="configs/experiments/main_experiment.yaml",
    )

    parser.add_argument("--algorithm", required=True)
    parser.add_argument("--distribution", required=True)
    parser.add_argument("--seed", type=int, required=True)

    args = parser.parse_args()

    config = load_experiment_config(args.config)
    validate_algorithm(args.algorithm, config)

    experiment = prepare_experiment(
        config=config,
        distribution=args.distribution,
        seed=args.seed,
    )
    result = run_selected_algorithm(
    algorithm=args.algorithm,
    experiment=experiment,
    config=config,
)

    history = result["history"]

    save_experiment_results(
    history=history,
    algorithm=args.algorithm,
    distribution=args.distribution,
    seed=args.seed,
    config=config,
)

    save_best_checkpoint(
        result=result,
        algorithm=args.algorithm,
        distribution=args.distribution,
        seed=args.seed,
    )

        # Restore the model with the best validation accuracy.
    experiment["server"].global_model.load_state_dict(
        result["best_model_state"]
    )

    # Evaluate the selected model on the untouched test set.
    test_metrics = experiment["server"].evaluate_global_model(
        experiment["test_loader"]
    )

    save_test_results(
    test_metrics=test_metrics,
    algorithm=args.algorithm,
    distribution=args.distribution,
    seed=args.seed,
    result=result,
    config=config,
)

    print("\nFinal Test Results")
    print(f"Test Loss: {test_metrics['loss']:.4f}")
    print(f"Test Accuracy: {test_metrics['accuracy']:.4f}")

    print(f"\nCompleted {len(history)} communication rounds.")

    print("\nExperiment setup successful")
    print(f"Algorithm: {args.algorithm}")
    print(f"Distribution: {args.distribution}")
    print(f"Seed: {args.seed}")
    print(f"Device: {experiment['device']}")
    print(f"Clients: {len(experiment['client_loaders'])}")

    print(
        f"Validation samples: "
        f"{len(experiment['validation_loader'].dataset)}"
    )

    print(
        f"Test samples: "
        f"{len(experiment['test_loader'].dataset)}"
    )

    print(f"Best round: {result['best_round']}")
    print(
        f"Best validation accuracy: "
        f"{result['best_validation_accuracy']:.4f}"
    )


if __name__ == "__main__":
    main()
