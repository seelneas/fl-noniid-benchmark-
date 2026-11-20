
import numpy as np
import torch

from torch.utils.data import TensorDataset

from flbench.data.datasets import (
    create_client_datasets,
    create_client_dataloaders,
    prepare_federated_data
)

def test_client_data_isolation():
    # Create a small synthetic dataset
    images = torch.randn(100, 3, 32, 32)
    labels = torch.randint(0, 10, (100,))

    dataset = TensorDataset(images, labels)

    # Divide 100 samples among 4 clients
    client_indices = {
        0: np.arange(0, 25),
        1: np.arange(25, 50),
        2: np.arange(50, 75),
        3: np.arange(75, 100),
    }

    client_datasets = create_client_datasets(
        dataset,
        client_indices,
        train_val_split=0.8,
        seed=42,
    )

    all_client_indices = []

    for client_id, datasets_dict in client_datasets.items():
        train = datasets_dict["train"]
        val = datasets_dict["val"]

        # Recover original dataset indices
        train_indices = {
            train.dataset.indices[i] for i in train.indices
        }

        val_indices = {
            val.dataset.indices[i] for i in val.indices
        }

        # Training and validation must not overlap
        assert train_indices.isdisjoint(val_indices)

        # Together they must cover the client's assigned samples
        assert train_indices | val_indices == set(
            client_indices[client_id]
        )

        all_client_indices.extend(
            train_indices | val_indices
        )

    # No sample should belong to multiple clients
    assert len(all_client_indices) == len(set(all_client_indices))

    # All original samples must be accounted for
    assert set(all_client_indices) == set(range(100))


def test_client_dataloaders():
    images = torch.randn(100, 3, 32, 32)
    labels = torch.randint(0, 10, (100,))

    dataset = TensorDataset(images, labels)

    client_indices = {
        0: np.arange(0, 50),
        1: np.arange(50, 100),
    }

    client_datasets = create_client_datasets(
        dataset,
        client_indices,
        seed=42,
    )

    loaders = create_client_dataloaders(
        client_datasets,
        batch_size=8,
    )

    assert len(loaders) == 2

    images, labels = next(iter(loaders[0]["train"]))

    assert images.shape == (8, 3, 32, 32)
    assert labels.shape == (8,)


def test_prepare_federated_data():
    client_loaders, test_loader, client_indices = prepare_federated_data(
        data_dir="data",
        num_clients=20,
        partition="dirichlet",
        alpha=0.5,
        batch_size=64,
        train_val_split=0.8,
        seed=42,
    )

    assert len(client_loaders) == 20
    assert len(test_loader.dataset) == 10000

    total_samples = sum(
        len(loaders["train"].dataset) + len(loaders["val"].dataset)
        for loaders in client_loaders.values()
    )

    assert total_samples == 50000

    all_indices = [
        int(index)
        for indices in client_indices.values()
        for index in indices
    ]

    assert len(all_indices) == len(set(all_indices))
    assert len(all_indices) == 50000
