
import numpy as np
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from flbench.data.partition import (
    dirichlet_partition,
    iid_partition,
)


def test_iid_partition():
    labels = np.repeat(np.arange(10), 5000)

    clients = iid_partition(
        labels,
        num_clients=20,
        seed=42,
    )

    assert len(clients) == 20
    assert all(len(indices) == 2500 for indices in clients.values())


def test_dirichlet_partition():
    labels = np.repeat(np.arange(10), 5000)

    clients = dirichlet_partition(
        labels,
        num_clients=20,
        alpha=0.5,
        seed=42,
    )

    assert len(clients) == 20

    all_indices = np.concatenate(list(clients.values()))

    assert len(all_indices) == len(labels)
    assert len(np.unique(all_indices)) == len(labels)
    assert all(len(indices) >= 10 for indices in clients.values())


def test_dirichlet_reproducibility():
    labels = np.repeat(np.arange(10), 5000)

    first = dirichlet_partition(labels, seed=42)
    second = dirichlet_partition(labels, seed=42)

    for client_id in first:
        np.testing.assert_array_equal(
            first[client_id],
            second[client_id],
        )
