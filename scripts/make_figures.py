
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from flbench.metrics import (
    get_distribution_label,
    get_distribution_order,
)


ALGORITHMS = [
    "fedavg",
    "fedprox",
    "scaffold",
]


def load_aggregated_results(results_path):
    """
    Load aggregated test results.
    """

    results_path = Path(results_path)

    if not results_path.exists():
        raise FileNotFoundError(
            f"Aggregated results not found: {results_path}"
        )

    results = pd.read_csv(results_path)

    required_columns = [
        "algorithm",
        "distribution",
        "mean_test_accuracy",
        "std_test_accuracy",
    ]

    missing = [
        column
        for column in required_columns
        if column not in results.columns
    ]

    if missing:
        raise ValueError(
            f"Aggregated results are missing columns: {missing}"
        )

    return results


def plot_algorithm_comparison(results, output_dir):
    """
    Plot mean test accuracy for each algorithm
    across all data distributions.
    """

    distributions = get_distribution_order(
        results["distribution"].unique().tolist()
    )

    algorithms = [
        algorithm
        for algorithm in ALGORITHMS
        if algorithm in results["algorithm"].unique()
    ]

    x = np.arange(len(distributions))
    width = 0.8 / max(len(algorithms), 1)

    fig, ax = plt.subplots(figsize=(12, 7))

    for index, algorithm in enumerate(algorithms):

        algorithm_data = (
            results[
                results["algorithm"] == algorithm
            ]
            .set_index("distribution")
            .reindex(distributions)
        )

        means = algorithm_data[
            "mean_test_accuracy"
        ].to_numpy()

        standard_deviations = algorithm_data[
            "std_test_accuracy"
        ].fillna(0).to_numpy()

        offset = (
            index - (len(algorithms) - 1) / 2
        ) * width

        ax.bar(
            x + offset,
            means,
            width,
            yerr=standard_deviations,
            capsize=4,
            label=algorithm.upper(),
        )

    ax.set_title(
        "Federated Learning Algorithm Comparison"
    )

    ax.set_xlabel("Data Distribution")
    ax.set_ylabel("Mean Test Accuracy")

    ax.set_xticks(x)

    ax.set_xticklabels(
        [
            get_distribution_label(distribution)
            for distribution in distributions
        ]
    )

    ax.set_ylim(0, 1)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()

    output_path = (
        output_dir / "algorithm_comparison.png"
    )

    fig.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)

    print(f"Saved: {output_path}")


def load_round_histories(results_dir, algorithm, distribution):
    """
    Load communication-round histories for all seeds
    belonging to one algorithm and distribution.
    """

    pattern = (
        f"{algorithm}_{distribution}_seed_*.csv"
    )

    files = sorted(
        results_dir.glob(pattern)
    )

    histories = []

    for file_path in files:

        # Ignore final test-result files.
        if file_path.stem.endswith("_test"):
            continue

        dataframe = pd.read_csv(file_path)

        required_columns = [
            "round",
            "validation_accuracy",
        ]

        if not all(
            column in dataframe.columns
            for column in required_columns
        ):
            print(
                f"Skipping incompatible history: {file_path.name}"
            )
            continue

        histories.append(dataframe)

    return histories


def plot_convergence(results_dir, output_dir):
    """
    Plot mean validation accuracy over communication rounds.

    Each panel represents one data distribution.
    """

    results_dir = Path(results_dir)

    distributions = get_distribution_order()

    fig, axes = plt.subplots(
        2,
        2,
        figsize=(14, 10),
        sharex=True,
        sharey=True,
    )

    axes = axes.flatten()

    plotted_any = False

    for ax, distribution in zip(
        axes,
        distributions,
    ):

        for algorithm in ALGORITHMS:

            histories = load_round_histories(
                results_dir,
                algorithm,
                distribution,
            )

            if not histories:
                continue

            combined = pd.concat(
                histories,
                ignore_index=True,
            )

            summary = (
                combined
                .groupby("round")["validation_accuracy"]
                .agg(["mean", "std"])
                .reset_index()
            )

            ax.plot(
                summary["round"],
                summary["mean"],
                label=algorithm.upper(),
            )

            plotted_any = True

        ax.set_title(
            get_distribution_label(distribution)
        )

        ax.set_xlabel("Communication Round")
        ax.set_ylabel("Validation Accuracy")

        ax.grid(alpha=0.3)
        ax.legend()

    if not plotted_any:
        plt.close(fig)
        print(
            "WARNING: No compatible round-history files found. "
            "Skipping convergence figure."
        )
        return

    fig.suptitle(
        "Federated Learning Convergence Across Data Distributions",
        fontsize=14,
    )

    fig.tight_layout()

    output_path = (
        output_dir / "convergence_comparison.png"
    )

    fig.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)

    print(f"Saved: {output_path}")


def plot_heterogeneity_impact(results, output_dir):
    """
    Plot test accuracy as client heterogeneity increases.
    """

    distributions = get_distribution_order(
        results["distribution"].unique().tolist()
    )

    algorithms = [
        algorithm
        for algorithm in ALGORITHMS
        if algorithm in results["algorithm"].unique()
    ]

    fig, ax = plt.subplots(figsize=(10, 7))

    x = np.arange(len(distributions))

    for algorithm in algorithms:

        algorithm_data = (
            results[
                results["algorithm"] == algorithm
            ]
            .set_index("distribution")
            .reindex(distributions)
        )

        means = algorithm_data[
            "mean_test_accuracy"
        ].to_numpy()

        standard_deviations = algorithm_data[
            "std_test_accuracy"
        ].fillna(0).to_numpy()

        ax.errorbar(
            x,
            means,
            yerr=standard_deviations,
            marker="o",
            capsize=4,
            linewidth=2,
            label=algorithm.upper(),
        )

    ax.set_title(
        "Impact of Client Heterogeneity on Test Accuracy"
    )

    ax.set_xlabel("Data Distribution")
    ax.set_ylabel("Mean Test Accuracy")

    ax.set_xticks(x)

    ax.set_xticklabels(
        [
            get_distribution_label(distribution)
            for distribution in distributions
        ]
    )

    ax.set_ylim(0, 1)
    ax.grid(alpha=0.3)
    ax.legend()

    fig.tight_layout()

    output_path = (
        output_dir / "heterogeneity_impact.png"
    )

    fig.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)

    print(f"Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate federated learning research figures."
    )

    parser.add_argument(
        "--results-dir",
        type=str,
        default="results/summaries",
        help="Directory containing experiment CSV files.",
    )

    parser.add_argument(
        "--aggregated-file",
        type=str,
        default="results/summaries/aggregated_results.csv",
        help="Path to aggregated results CSV.",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/figures",
        help="Directory for generated figures.",
    )

    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    output_dir = Path(args.output_dir)

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 65)
    print("FEDERATED LEARNING RESEARCH VISUALIZATIONS")
    print("=" * 65)

    results = load_aggregated_results(
        args.aggregated_file
    )

    plot_algorithm_comparison(
        results,
        output_dir,
    )

    plot_convergence(
        results_dir,
        output_dir,
    )

    plot_heterogeneity_impact(
        results,
        output_dir,
    )

    print("\nFigure generation completed.")


if __name__ == "__main__":
    main()
