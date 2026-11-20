import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from flbench.models.cnn import SimpleCNN
from flbench.fl.server import FederatedServer
from flbench.algorithms.fedprox_simulation import run_fedprox_simulation


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


def test_fedprox_simulation_runs(simulation_setup):
    server, client_loaders, test_loader = simulation_setup

    metrics = run_fedprox_simulation(
        server=server,
        client_loaders=client_loaders,
        test_loader=test_loader,
        num_rounds=2,
        local_epochs=1,
        learning_rate=0.01,
        mu=0.01,
    )

    assert len(metrics) == 2
    assert metrics[0]["round"] == 1
    assert metrics[1]["round"] == 2

    assert "train_loss" in metrics[0]
    assert "test_accuracy" in metrics[0]
    assert metrics[0]["mu"] == 0.01


def test_fedprox_simulation_rejects_invalid_mu(simulation_setup):
    server, client_loaders, test_loader = simulation_setup

    with pytest.raises(ValueError):
        run_fedprox_simulation(
            server,
            client_loaders,
            test_loader,
            num_rounds=1,
            local_epochs=1,
            mu=-0.1,
        )


def test_fedprox_simulation_rejects_invalid_rounds(simulation_setup):
    server, client_loaders, test_loader = simulation_setup

    with pytest.raises(ValueError):
        run_fedprox_simulation(
            server,
            client_loaders,
            test_loader,
            num_rounds=0,
        )