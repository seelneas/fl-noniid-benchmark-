
import os

import matplotlib.pyplot as plt
import numpy as np

from flbench.data.datasets import download_cifar10
from flbench.data.partition import (
    dirichlet_partition,
    get_client_class_distribution,
    iid_partition,
)


def main():
    # Reproducible experiment settings
    NUM_CLIENTS = 20
    SEED = 42

    # Load CIFAR-10
    train_dataset, _ = download_cifar10()
    labels = np.array(train_dataset.targets)
    class_names = train_dataset.classes

    # Generate four partition configurations
    partitions = {
        "IID": iid_partition(
            labels,
            num_clients=NUM_CLIENTS,
            seed=SEED,
        ),
        "Mild (alpha=10.0)": dirichlet_partition(
            labels,
            num_clients=NUM_CLIENTS,
            alpha=10.0,
            seed=SEED,
        ),
        "Moderate (alpha=0.5)": dirichlet_partition(
            labels,
            num_clients=NUM_CLIENTS,
            alpha=0.5,
            seed=SEED,
        ),
        "Severe (alpha=0.1)": dirichlet_partition(
            labels,
            num_clients=NUM_CLIENTS,
            alpha=0.1,
            seed=SEED,
        ),
    }

    # Prepare visualization
    fig, axes = plt.subplots(2, 2, figsize=(18, 12))

    for ax, (name, partition) in zip(axes.flat, partitions.items()):

        distribution = get_client_class_distribution(
            labels,
            partition,
        )

        matrix = np.array([
            distribution[client_id]
            for client_id in range(NUM_CLIENTS)
        ])

        image = ax.imshow(
            matrix,
            aspect="auto",
            cmap="Blues",
        )

        ax.set_title(name, fontsize=13)
        ax.set_xlabel("CIFAR-10 Classes")
        ax.set_ylabel("Clients")

        ax.set_xticks(range(len(class_names)))
        ax.set_xticklabels(
            class_names,
            rotation=45,
            ha="right",
        )

        ax.set_yticks(range(NUM_CLIENTS))
        ax.set_yticklabels(range(NUM_CLIENTS))

        fig.colorbar(
            image,
            ax=ax,
            label="Number of samples",
        )

        print(f"\n{name}")
        print("-" * 40)

        client_sizes = matrix.sum(axis=1)

        print("Minimum client samples:", client_sizes.min())
        print("Maximum client samples:", client_sizes.max())
        print("Average client samples:", client_sizes.mean())

    fig.suptitle(
        "CIFAR-10 Client Class Distributions",
        fontsize=17,
    )

    plt.tight_layout()

    # Save visualization
    output_dir = "results/figures"
    os.makedirs(output_dir, exist_ok=True)

    output_path = os.path.join(
        output_dir,
        "client_class_distributions.png",
    )

    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )

    print(f"\nVisualization saved to: {output_path}")

    plt.show()


if __name__ == "__main__":
    main()
