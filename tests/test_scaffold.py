import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from flbench.models.cnn import SimpleCNN
from flbench.algorithms.scaffold import train_scaffold_model


@pytest.fixture
def scaffold_setup():
    model = SimpleCNN(num_classes=10)

    images = torch.randn(16, 3, 32, 32)
    labels = torch.randint(0, 10, (16,))

    dataset = TensorDataset(images, labels)
    loader = DataLoader(dataset, batch_size=8)

    client_control = [
        torch.zeros_like(parameter)
        for parameter in model.parameters()
    ]

    server_control = [
        torch.zeros_like(parameter)
        for parameter in model.parameters()
    ]

    return model, loader, client_control, server_control


def test_scaffold_training(scaffold_setup):
    model, loader, client_control, server_control = scaffold_setup

    local_model, metrics = train_scaffold_model(
        global_model=model,
        train_loader=loader,
        client_control=client_control,
        server_control=server_control,
        device=torch.device("cpu"),
        local_epochs=1,
    )

    assert local_model is not model
    assert 0 <= metrics["train_accuracy"] <= 1
    assert metrics["train_loss"] >= 0


def test_scaffold_does_not_modify_global_model(scaffold_setup):
    model, loader, client_control, server_control = scaffold_setup

    original_parameters = [
        parameter.detach().clone()
        for parameter in model.parameters()
    ]

    train_scaffold_model(
        global_model=model,
        train_loader=loader,
        client_control=client_control,
        server_control=server_control,
        device=torch.device("cpu"),
        local_epochs=1,
    )

    for original, current in zip(
        original_parameters,
        model.parameters(),
    ):
        assert torch.equal(original, current)


def test_scaffold_zero_controls(scaffold_setup):
    model, loader, client_control, server_control = scaffold_setup

    local_model, metrics = train_scaffold_model(
        global_model=model,
        train_loader=loader,
        client_control=client_control,
        server_control=server_control,
        device=torch.device("cpu"),
        local_epochs=1,
    )

    assert local_model is not None
    assert torch.isfinite(torch.tensor(metrics["train_loss"]))


def test_scaffold_rejects_invalid_epochs(scaffold_setup):
    model, loader, client_control, server_control = scaffold_setup

    with pytest.raises(ValueError):
        train_scaffold_model(
            global_model=model,
            train_loader=loader,
            client_control=client_control,
            server_control=server_control,
            device=torch.device("cpu"),
            local_epochs=0,
        )