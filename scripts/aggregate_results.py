
import argparse
from pathlib import Path

import pandas as pd

from flbench.metrics import aggregate_seed_metrics


REQUIRED_COLUMNS = [
    "algorithm",
    "distribution",
    "seed",
    "best_round",
    "best_validation_accuracy",
    "test_loss",
    "test_accuracy",
]


def load_test_results(results_dir):
    """
    Load final test-result CSV files from the results directory.
    """

    results_dir = Path(results_dir)

    if not results_dir.exists():
        raise FileNotFoundError(
            f"Results directory does not exist: {results_dir}"
        )

    result_files = sorted(
        results_dir.glob("*_test.csv")
    )

    if not result_files:
        raise FileNotFoundError(
            f"No test-result CSV files found in {results_dir}"
        )

    frames = []

    for file_path in result_files:
        print(f"Loading: {file_path.name}")

        dataframe = pd.read_csv(file_path)

        missing = [
            column
            for column in REQUIRED_COLUMNS
            if column not in dataframe.columns
        ]

        if missing:
            raise ValueError(
                f"{file_path.name} is missing columns: {missing}"
            )

        frames.append(dataframe)

    results = pd.concat(
        frames,
        ignore_index=True,
    )

    return results


def validate_experiment_results(results):
    """
    Check for duplicate experiment records and invalid values.
    """

    if results.empty:
        raise ValueError("No experiment results were loaded.")

    results = results.copy()

    results["seed"] = pd.to_numeric(
        results["seed"],
        errors="raise",
    )

    duplicate_columns = [
        "algorithm",
        "distribution",
        "seed",
    ]

    duplicates = results.duplicated(
        subset=duplicate_columns,
        keep=False,
    )

    if duplicates.any():
        duplicate_records = results.loc[
            duplicates,
            duplicate_columns,
        ]

        raise ValueError(
            "Duplicate experiment records detected:\n"
            f"{duplicate_records.to_string(index=False)}"
        )

    for column in [
        "test_accuracy",
        "test_loss",
        "best_validation_accuracy",
    ]:
        results[column] = pd.to_numeric(
            results[column],
            errors="raise",
        )

    if not results[
        ["test_accuracy", "test_loss", "best_validation_accuracy"]
    ].notna().all().all():
        raise ValueError(
            "Experiment results contain missing metric values."
        )

    return results


def save_results(results, aggregated, output_dir):
    """
    Save combined and aggregated experiment results.
    """

    output_dir = Path(output_dir)
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    raw_path = output_dir / "all_test_results.csv"

    aggregated_path = (
        output_dir / "aggregated_results.csv"
    )

    results.to_csv(
        raw_path,
        index=False,
    )

    aggregated.to_csv(
        aggregated_path,
        index=False,
    )

    print("\nResults saved successfully.")

    print(f"Combined results: {raw_path}")
    print(f"Aggregated results: {aggregated_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Aggregate federated learning experiment results."
    )

    parser.add_argument(
        "--input-dir",
        type=str,
        default="results/summaries",
        help="Directory containing final test-result CSV files.",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/summaries",
        help="Directory where aggregated results will be saved.",
    )

    parser.add_argument(
        "--expected-runs",
        type=int,
        default=36,
        help="Expected number of completed experiments.",
    )

    args = parser.parse_args()

    print("=" * 65)
    print("FEDERATED LEARNING RESULTS AGGREGATION")
    print("=" * 65)

    results = load_test_results(
        args.input_dir
    )

    results = validate_experiment_results(
        results
    )

    actual_runs = len(results)

    print(f"\nExperiments found: {actual_runs}")
    print(f"Expected experiments: {args.expected_runs}")

    if actual_runs != args.expected_runs:
        print(
            "\nWARNING: The number of result files does not "
            "match the expected number of experiments."
        )

    print("\nExperiments per algorithm:")

    print(
        results.groupby("algorithm")["seed"]
        .count()
        .to_string()
    )

    print("\nAggregating results across seeds...")

    aggregated = aggregate_seed_metrics(
        results
    )

    save_results(
        results,
        aggregated,
        args.output_dir,
    )

    print("\nAggregated results:")
    print(
        aggregated.to_string(
            index=False,
            float_format=lambda value: f"{value:.4f}",
        )
    )

    print("\nAggregation completed.")


if __name__ == "__main__":
    main()
