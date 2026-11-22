
import torch

from torch.utils.data import DataLoader, TensorDataset

from flbench.models.cnn import SimpleCNN
from flbench.fl.server import FederatedServer
from flbench.fl.simulation import run_federated_simulation


def create_dummy_loader(num_samples=16):
    images = torch.randn(num_samples, 3, 32, 32)
    labels = torch.randint(0, 10, (num_samples,))

    dataset = TensorDataset(images, labels)

    return DataLoader(
        dataset,
        batch_size=8,
        shuffle=False,
    )


def test_federated_simulation_runs():
    device = torch.device("cpu")

    server = FederatedServer(
        SimpleCNN(),
        device,
    )

    client_loaders = {
        0: {
            "train": create_dummy_loader(16),
            "val": create_dummy_loader(4),
        },
        1: {
            "train": create_dummy_loader(24),
            "val": create_dummy_loader(4),
        },
    }

    validation_loader = create_dummy_loader(16)

    results = run_federated_simulation(
        server=server,
        client_loaders=client_loaders,
        validation_loader=validation_loader,
        num_rounds=2,
        local_epochs=1,
        learning_rate=0.01,
    )

    # Check the list of round metrics inside "history"
    assert len(results["history"]) == 2

    assert results["history"][0]["round"] == 1
    assert results["history"][1]["round"] == 2

    assert results["history"][0]["num_clients"] == 2
    assert results["history"][0]["total_client_samples"] == 40

    assert 0 <= results["history"][0]["validation_accuracy"] <= 1
    assert results["history"][0]["validation_loss"] >= 0

    # Check the best model tracking metadata
    assert results["best_model_state"] is not None
    assert results["best_round"] in [1, 2]
    assert 0 <= results["best_validation_accuracy"] <= 1

def test_simulation_invalid_rounds():
    server = FederatedServer(
        SimpleCNN(),
        torch.device("cpu"),
    )

    with torch.no_grad():
        loader = create_dummy_loader(16)

        try:
            run_federated_simulation(
                server=server,
                client_loaders={0: {"train": loader}},
                validation_loader=loader,
                num_rounds=0,
            )
        except ValueError:
            pass
        else:
            raise AssertionError(
                "Expected ValueError for zero communication rounds"
            )


def test_simulation_rejects_empty_clients():
    server = FederatedServer(
        SimpleCNN(),
        torch.device("cpu"),
    )

    loader = create_dummy_loader(16)

    try:
        run_federated_simulation(
            server=server,
            client_loaders={},
            validation_loader=loader,
        )
    except ValueError:
        pass
    else:
        raise AssertionError(
            "Expected ValueError for empty client collection"
        )
