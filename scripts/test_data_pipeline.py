
from flbench.data.datasets import prepare_federated_data

def main():
    client_loaders, test_loader, client_indices = prepare_federated_data(
        data_dir="data",
        num_clients=20,
        partition="dirichlet",
        alpha=0.5,
        batch_size=64,
        train_val_split=0.8,
        seed=42,
    )

    print("\nFEDERATED DATA PIPELINE SUMMARY")
    print("=" * 45)

    print(f"Number of clients: {len(client_loaders)}")

    total_train = 0
    total_val = 0

    for client_id, loaders in client_loaders.items():
        train_size = len(loaders["train"].dataset)
        val_size = len(loaders["val"].dataset)

        total_train += train_size
        total_val += val_size

        print(
            f"Client {client_id:02d}: "
            f"Train={train_size}, Validation={val_size}"
        )

    print("\nDATA TOTALS")
    print(f"Client training samples: {total_train}")
    print(f"Client validation samples: {total_val}")
    print(f"Total client samples: {total_train + total_val}")
    print(f"Global test samples: {len(test_loader.dataset)}")

    print("\nBATCH VERIFICATION")

    images, labels = next(iter(client_loaders[0]["train"]))

    print(f"Image shape: {images.shape}")
    print(f"Label shape: {labels.shape}")

    assert len(client_loaders) == 20
    assert total_train + total_val == 50000
    assert len(test_loader.dataset) == 10000
    assert images.shape[1:] == (3, 32, 32)

    print("\nAll pipeline checks passed!")


if __name__ == "__main__":
    main()
