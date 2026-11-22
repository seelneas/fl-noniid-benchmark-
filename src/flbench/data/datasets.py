
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset, random_split, ConcatDataset
from torchvision import datasets, transforms


def download_cifar10(data_dir: str = "data"):
    """
    Download CIFAR-10 training and test datasets.
    """
    data_path = Path(data_dir)
    data_path.mkdir(parents=True, exist_ok=True)

    transform = transforms.ToTensor()

    train_dataset = datasets.CIFAR10(
        root=str(data_path),
        train=True,
        download=True,
        transform=transform,
    )

    test_dataset = datasets.CIFAR10(
        root=str(data_path),
        train=False,
        download=True,
        transform=transform,
    )

    return train_dataset, test_dataset


def create_client_datasets(
    train_dataset,
    client_indices,
    train_val_split: float = 0.8,
    seed: int = 42,
):
    """
    Create independent training and validation datasets
    for every federated client.

    Args:
        train_dataset: Complete CIFAR-10 training dataset.
        client_indices: Dictionary mapping client IDs to indices.
        train_val_split: Fraction used for local training.
        seed: Random seed for reproducibility.

    Returns:
        Dictionary containing client training and validation datasets.
    """
    if not 0 < train_val_split < 1:
        raise ValueError("train_val_split must be between 0 and 1.")

    client_datasets = {}

    for client_id, indices in client_indices.items():
        indices = np.asarray(indices)

        if len(indices) < 2:
            raise ValueError(
                f"Client {client_id} needs at least 2 samples."
            )

        train_size = int(len(indices) * train_val_split)
        val_size = len(indices) - train_size

        generator = torch.Generator().manual_seed(
            seed + int(client_id)
        )

        train_subset, val_subset = random_split(
            Subset(train_dataset, indices.tolist()),
            [train_size, val_size],
            generator=generator,
        )

        client_datasets[client_id] = {
            "train": train_subset,
            "val": val_subset,
        }

    return client_datasets


def create_client_dataloaders(
    client_datasets,
    batch_size: int = 64,
    seed: int = 42,
):
    """
    Create independent PyTorch DataLoaders for every client.
    """
    if batch_size < 1:
        raise ValueError("batch_size must be positive.")

    client_loaders = {}

    for client_id, datasets_dict in client_datasets.items():
        generator = torch.Generator().manual_seed(
            seed + int(client_id)
        )

        train_loader = DataLoader(
            datasets_dict["train"],
            batch_size=batch_size,
            shuffle=True,
            generator=generator,
        )

        val_loader = DataLoader(
            datasets_dict["val"],
            batch_size=batch_size,
            shuffle=False,
        )

        client_loaders[client_id] = {
            "train": train_loader,
            "val": val_loader,
        }

    return client_loaders


def create_test_dataloader(
    test_dataset,
    batch_size: int = 64,
):
    """
    Create a global test DataLoader.
    """
    return DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
    )


def prepare_federated_data(
    data_dir="data",
    num_clients=20,
    partition="dirichlet",
    alpha=0.5,
    batch_size=64,
    train_val_split=0.8,
    seed=42,
):
    """
    Prepare CIFAR-10 datasets and DataLoaders for federated learning.

    Returns:
        client_loaders: Training and validation DataLoaders for each client.
        test_loader: Global test DataLoader.
        client_indices: Original training indices assigned to each client.
    """

    from flbench.data.partition import dirichlet_partition, iid_partition

    # 1. Download and load CIFAR-10
    train_dataset, test_dataset = download_cifar10(data_dir)

    # 2. Partition the training dataset
    labels = np.array(train_dataset.targets)

    if partition == "iid":
        client_indices = iid_partition(
            labels,
            num_clients=num_clients,
            seed=seed,
        )

    elif partition == "dirichlet":
        client_indices = dirichlet_partition(
            labels,
            num_clients=num_clients,
            alpha=alpha,
            seed=seed,
        )

    else:
        raise ValueError(
            "partition must be either 'iid' or 'dirichlet'"
        )

    # 3. Create client-specific training and validation datasets
    client_datasets = create_client_datasets(
        train_dataset,
        client_indices,
        train_val_split=train_val_split,
        seed=seed,
    )

    # 4. Create client DataLoaders
    client_loaders = create_client_dataloaders(
        client_datasets,
        batch_size=batch_size,
        seed=seed,
    )


    # 5. Create the global test DataLoader
    test_loader = create_test_dataloader(
        test_dataset,
        batch_size=batch_size,
    )

    validation_loader = create_global_validation_dataloader(
    client_datasets,
    batch_size=batch_size,
)

    return client_loaders, test_loader, client_indices, validation_loader


def create_global_validation_dataloader(
    client_datasets,
    batch_size=64,
):
    validation_datasets = [
        datasets_dict["val"]
        for datasets_dict in client_datasets.values()
    ]

    global_validation_dataset = ConcatDataset(
        validation_datasets
    )

    return DataLoader(
        global_validation_dataset,
        batch_size=batch_size,
        shuffle=False,
    )