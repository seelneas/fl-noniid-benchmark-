
import torch
from torch.utils.data import DataLoader, TensorDataset

from flbench.models.cnn import SimpleCNN
from flbench.algorithms.fedprox import train_fedprox_model


def create_dummy_loader():
    images = torch.randn(16, 3, 32, 32)
    labels = torch.randint(0, 10, (16,))

    dataset = TensorDataset(images, labels)

    return DataLoader(
        dataset,
        batch_size=8,
        shuffle=False,
    )


def test_fedprox_training():
    model = SimpleCNN()
    loader = create_dummy_loader()

    local_model, metrics = train_fedprox_model(
        global_model=model,
        train_loader=loader,
        device=torch.device("cpu"),
        local_epochs=1,
        mu=0.01,
    )

    assert local_model is not model
    assert metrics["train_loss"] >= 0
    assert 0 <= metrics["train_accuracy"] <= 1


def test_fedprox_does_not_modify_global_model():
    model = SimpleCNN()
    loader = create_dummy_loader()

    original_parameters = {
        name: parameter.detach().clone()
        for name, parameter in model.named_parameters()
    }

    train_fedprox_model(
        global_model=model,
        train_loader=loader,
        device=torch.device("cpu"),
        local_epochs=1,
        mu=0.01,
    )

    for name, parameter in model.named_parameters():
        assert torch.equal(
            parameter,
            original_parameters[name],
        )


def test_fedprox_zero_mu():
    model = SimpleCNN()
    loader = create_dummy_loader()

    local_model, metrics = train_fedprox_model(
        global_model=model,
        train_loader=loader,
        device=torch.device("cpu"),
        local_epochs=1,
        mu=0.0,
    )

    assert local_model is not model
    assert metrics["train_loss"] >= 0


def test_fedprox_rejects_negative_mu():
    model = SimpleCNN()
    loader = create_dummy_loader()

    try:
        train_fedprox_model(
            global_model=model,
            train_loader=loader,
            device=torch.device("cpu"),
            mu=-0.1,
        )
    except ValueError:
        pass
    else:
        raise AssertionError("Expected ValueError for negative mu")
