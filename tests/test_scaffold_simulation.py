import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from flbench.models.cnn import SimpleCNN
from flbench.fl.server import FederatedServer
from flbench.algorithms.scaffold_simulation import (
    run_scaffold_simulation,
)


@pytest.fixture
def simulation_setup():
    device = torch.device("cpu")

    client_loaders = {}

    for client_id in range(2):
        images = torch.randn(16, 3, 32, 32)
        labels = torch.randint(0, 10, (16,))

        dataset = TensorDataset(images, labels)

        loader = DataLoader(
            dataset,
            batch_size=8,
            shuffle=True,
        )

        client_loaders[client_id] = {
            "train": loader
        }

    test_images = torch.randn(12, 3, 32, 32)
    test_labels = torch.randint(0, 10, (12,))

    test_dataset = TensorDataset(test_images, test_labels)

    test_loader = DataLoader(
        test_dataset,
        batch_size=6,
    )

    model = SimpleCNN(num_classes=10)
    server = FederatedServer(model, device)

    return server, client_loaders, test_loader


def test_scaffold_simulation_runs(simulation_setup):
    server, client_loaders, test_loader = simulation_setup

    results = run_scaffold_simulation(
        server=server,
        client_loaders=client_loaders,
        validation_loader=test_loader,
        num_rounds=2,
        local_epochs=1,
        learning_rate=0.01,
    )

    assert len(results['history']) == 2
    assert results['history'][0]["round"] == 1
    assert results['history'][1]["round"] == 2

    assert "validation_loss" in results['history'][0]
    assert "validation_accuracy" in results['history'][0]
    assert results["best_model_state"] is not None
    assert results["best_round"] in [1, 2]
    assert 0.0 <= results["best_validation_accuracy"] <= 1.0


def test_scaffold_simulation_rejects_invalid_rounds(simulation_setup):
    server, client_loaders, test_loader = simulation_setup

    with pytest.raises(ValueError):
        run_scaffold_simulation(
            server,
            client_loaders,
            test_loader,
            num_rounds=0,
        )


def test_scaffold_simulation_rejects_invalid_epochs(simulation_setup):
    server, client_loaders, test_loader = simulation_setup

    with pytest.raises(ValueError):
        run_scaffold_simulation(
            server,
            client_loaders,
            test_loader,
            local_epochs=0,
        )


def test_scaffold_simulation_rejects_invalid_learning_rate(simulation_setup):
    server, client_loaders, test_loader = simulation_setup

    with pytest.raises(ValueError):
        run_scaffold_simulation(
            server,
            client_loaders,
            test_loader,
            learning_rate=0,
        )