
from typing import Sequence

import pandas as pd


DEFAULT_METRICS = [
    "test_accuracy",
    "test_loss",
    "best_validation_accuracy",
]


DISTRIBUTION_ORDER = [
    "iid",
    "dirichlet_alpha_10",
    "dirichlet_alpha_05",
    "dirichlet_alpha_01",
]


DISTRIBUTION_LABELS = {
    "iid": "IID",
    "dirichlet_alpha_10": "Dirichlet α=10",
    "dirichlet_alpha_05": "Dirichlet α=0.5",
    "dirichlet_alpha_01": "Dirichlet α=0.1",
}


def validate_results_dataframe(
    results: pd.DataFrame,
    required_columns: Sequence[str],
) -> None:
    """
    Validate that a results DataFrame contains required columns.
    """
    missing_columns = [
        column
        for column in required_columns
        if column not in results.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )

    if results.empty:
        raise ValueError("The results DataFrame is empty.")


def aggregate_seed_metrics(
    results: pd.DataFrame,
    group_columns=None,
    metric_columns=None,
) -> pd.DataFrame:
    """
    Aggregate experiment results across random seeds.

    Groups results by algorithm and data distribution.

    Calculates:
        - Number of seeds
        - Mean of each metric
        - Standard deviation of each metric

    Standard deviation uses sample standard deviation (ddof=1).
    """

    if group_columns is None:
        group_columns = ["algorithm", "distribution"]

    if metric_columns is None:
        metric_columns = DEFAULT_METRICS

    required_columns = list(
    dict.fromkeys(
        list(group_columns) + list(metric_columns) + ["seed"]
    )
)

    validate_results_dataframe(
        results,
        required_columns,
    )

    results = results.copy()

    for column in metric_columns:
        results[column] = pd.to_numeric(
            results[column],
            errors="coerce",
        )

    if results[metric_columns].isna().any().any():
        raise ValueError(
            "One or more metric columns contain missing or "
            "non-numeric values."
        )

    aggregation = {}

    for metric in metric_columns:
        aggregation[f"mean_{metric}"] = (
            metric,
            "mean",
        )

        aggregation[f"std_{metric}"] = (
            metric,
            "std",
        )

    aggregation["num_seeds"] = (
        "seed",
        "nunique",
    )

    aggregated = (
        results
        .groupby(group_columns, as_index=False)
        .agg(**aggregation)
    )

    return aggregated


def calculate_accuracy_drop(
    aggregated_results: pd.DataFrame,
) -> pd.DataFrame:
    """
    Calculate accuracy reduction relative to IID performance.

    Accuracy drop is expressed in percentage points.

    Positive value:
        Accuracy is lower than IID.

    Negative value:
        Accuracy is higher than IID.
    """

    validate_results_dataframe(
        aggregated_results,
        [
            "algorithm",
            "distribution",
            "mean_test_accuracy",
        ],
    )

    results = aggregated_results.copy()

    iid_results = (
        results[
            results["distribution"] == "iid"
        ][["algorithm", "mean_test_accuracy"]]
        .rename(
            columns={
                "mean_test_accuracy": "iid_accuracy"
            }
        )
    )

    results = results.merge(
        iid_results,
        on="algorithm",
        how="left",
    )

    results["accuracy_drop_percentage_points"] = (
        results["iid_accuracy"]
        - results["mean_test_accuracy"]
    ) * 100

    return results


def get_distribution_label(distribution: str) -> str:
    """
    Convert a distribution identifier into a readable label.
    """

    return DISTRIBUTION_LABELS.get(
        distribution,
        distribution,
    )


def get_distribution_order(distributions=None):
    """
    Return distributions in the intended heterogeneity order.
    """

    if distributions is None:
        distributions = DISTRIBUTION_ORDER

    ordered = [
        distribution
        for distribution in DISTRIBUTION_ORDER
        if distribution in distributions
    ]

    remaining = [
        distribution
        for distribution in distributions
        if distribution not in ordered
    ]

    return ordered + remaining
