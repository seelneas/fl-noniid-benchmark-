
import yaml
import pandas as pd
from pathlib import Path

from flbench.models.cnn import SimpleCNN
from flbench.fl.server import FederatedServer
from flbench.fl.simulation import run_federated_simulation
from flbench.data.datasets import prepare_federated_data
from flbench.utils import set_seed, get_device


def main():

    # 1. Load configuration
    with open("configs/base.yaml", "r") as file:
        config = yaml.safe_load(file)

    # 2. Set reproducibility
    seed = config["seed"]
    set_seed(seed)

    # 3. Select device
    device = get_device()
    print(f"Using device: {device}")

    # 4. Prepare federated data
    print("\nPreparing federated datasets...")

    client_loaders, test_loader, client_indices, validation_loader = (
        prepare_federated_data(
            data_dir="data",
            num_clients=config["data"]["num_clients"],
            partition="dirichlet",
            alpha=0.5,
            batch_size=config["data"]["batch_size"],
            train_val_split=config["data"]["train_val_split"],
            seed=seed,
        )
    )

    print(f"Number of clients: {len(client_loaders)}")
    print(f"Global test samples: {len(test_loader.dataset)}")

    # 5. Initialize global model
    model = SimpleCNN(
        num_classes=config["model"]["num_classes"]
    )

    server = FederatedServer(
        global_model=model,
        device=device,
    )

    # 6. Run federated simulation
    print("\nStarting FedAvg experiment...")

    history = run_federated_simulation(
        server=server,
        client_loaders=client_loaders,
        validation_loader=validation_loader,
        num_rounds=config["federated"]["num_rounds"],
        local_epochs=config["federated"]["local_epochs"],
        learning_rate=config["optimizer"]["learning_rate"],
        momentum=config["optimizer"]["momentum"],
        weight_decay=config["optimizer"]["weight_decay"],
    )

    # 7. Save results
    results_dir = Path("results/summaries")
    results_dir.mkdir(parents=True, exist_ok=True)

    results_df = pd.DataFrame(history)

    results_path = results_dir / "fedavg_pilot.csv"
    results_df.to_csv(results_path, index=False)

    # 8. Save global model checkpoint
    checkpoint_dir = Path("checkpoints")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    torch_path = checkpoint_dir / "fedavg_pilot.pt"

    import torch
    torch.save(server.global_model.state_dict(), torch_path)

    print("\nEXPERIMENT COMPLETED")
    print("=" * 45)

    print(results_df.to_string(index=False))

    print(f"\nResults saved to: {results_path}")
    print(f"Model saved to: {torch_path}")


if __name__ == "__main__":
    main()
