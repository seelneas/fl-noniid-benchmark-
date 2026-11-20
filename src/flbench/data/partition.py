
from collections import defaultdict

import numpy as np


def dirichlet_partition(
    labels,
    num_clients: int = 20,
    alpha: float = 0.5,
    seed: int = 42,
    min_samples: int = 10,
):
    """
    Partition a dataset among clients using a Dirichlet distribution.

    Args:
        labels: Class labels for the complete dataset.
        num_clients: Number of federated clients.
        alpha: Dirichlet concentration parameter.
        seed: Random seed.
        min_samples: Minimum samples required per client.

    Returns:
        Dictionary mapping client IDs to sample indices.
    """
    if num_clients < 1:
        raise ValueError("num_clients must be positive.")

    if alpha <= 0:
        raise ValueError("alpha must be positive.")

    if min_samples < 0:
        raise ValueError("min_samples cannot be negative.")

    labels = np.asarray(labels)
    rng = np.random.default_rng(seed)

    num_classes = len(np.unique(labels))

    while True:
        client_indices = defaultdict(list)

        for class_id in np.unique(labels):
            class_indices = np.where(labels == class_id)[0]
            rng.shuffle(class_indices)

            proportions = rng.dirichlet(
                np.repeat(alpha, num_clients)
            )

            # Balance the number of samples assigned to clients.
            proportions = np.array([
                p * (len(client_indices[i]) < len(labels) / num_clients)
                for i, p in enumerate(proportions)
            ])

            proportions = proportions / proportions.sum()

            split_points = (
                np.cumsum(proportions) * len(class_indices)
            ).astype(int)[:-1]

            client_splits = np.split(class_indices, split_points)

            for client_id, indices in enumerate(client_splits):
                client_indices[client_id].extend(indices.tolist())

        sizes = [
            len(client_indices[i]) for i in range(num_clients)
        ]

        if min(sizes) >= min_samples:
            break

    result = {}

    for client_id in range(num_clients):
        indices = np.array(client_indices[client_id])
        rng.shuffle(indices)
        result[client_id] = indices

    return result


def iid_partition(
    labels,
    num_clients: int = 20,
    seed: int = 42,
):
    """
    Partition a dataset randomly and approximately equally.
    """
    labels = np.asarray(labels)
    rng = np.random.default_rng(seed)

    indices = np.arange(len(labels))
    rng.shuffle(indices)

    splits = np.array_split(indices, num_clients)

    return {
        client_id: split
        for client_id, split in enumerate(splits)
    }


def get_client_class_distribution(labels, client_indices):
    """
    Calculate the class distribution for each client.
    """
    labels = np.asarray(labels)
    distribution = {}

    for client_id, indices in client_indices.items():
        counts = np.bincount(
            labels[indices],
            minlength=10,
        )

        distribution[client_id] = counts

    return distribution
