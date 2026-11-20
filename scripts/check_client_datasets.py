
from flbench.data.datasets import (
    create_client_dataloaders,
    create_client_datasets,
    create_test_dataloader,
    download_cifar10,
)
from flbench.data.partition import (
    dirichlet_partition,
)

# Experiment settings
NUM_CLIENTS = 20
ALPHA = 0.5
SEED = 42
BATCH_SIZE = 64

# Load CIFAR-10
train_dataset, test_dataset = download_cifar10()

labels = train_dataset.targets

# Generate a non-IID partition
client_indices = dirichlet_partition(
    labels,
    num_clients=NUM_CLIENTS,
    alpha=ALPHA,
    seed=SEED,
)

# Create individual client datasets
client_datasets = create_client_datasets(
    train_dataset,
    client_indices,
    train_val_split=0.8,
    seed=SEED,
)

# Create client DataLoaders
client_loaders = create_client_dataloaders(
    client_datasets,
    batch_size=BATCH_SIZE,
    seed=SEED,
)

# Create global test DataLoader
test_loader = create_test_dataloader(
    test_dataset,
    batch_size=BATCH_SIZE,
)

# Print client statistics
print("\nCLIENT DATASET SUMMARY")
print("=" * 45)

for client_id, loaders in client_loaders.items():
    train_size = len(loaders["train"].dataset)
    val_size = len(loaders["val"].dataset)

    print(
        f"Client {client_id:02d}: "
        f"Train={train_size:4d}, "
        f"Validation={val_size:4d}"
    )

print("\nGLOBAL TEST DATASET")
print("=" * 45)
print("Test samples:", len(test_loader.dataset))

# Check one batch
images, targets = next(iter(client_loaders[0]["train"]))

print("\nSAMPLE TRAINING BATCH")
print("=" * 45)
print("Image batch shape:", images.shape)
print("Target batch shape:", targets.shape)
print("Number of clients:", len(client_loaders))
