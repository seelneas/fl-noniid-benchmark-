
import torch
from torch.utils.data import DataLoader, TensorDataset

from flbench.models.cnn import SimpleCNN
from flbench.fl.client import train_local_model, evaluate_model


def create_dummy_loader():
    images = torch.randn(16, 3, 32, 32)
    labels = torch.randint(0, 10, (16,))

    dataset = TensorDataset(images, labels)

    return DataLoader(
        dataset,
        batch_size=8,
        shuffle=False,
    )


def test_local_training():
    model = SimpleCNN(num_classes=10)
    loader = create_dummy_loader()

    device = torch.device("cpu")

    local_model, metrics = train_local_model(
        global_model=model,
        train_loader=loader,
        device=device,
        local_epochs=1,
        learning_rate=0.01,
    )

    assert local_model is not model

    assert 0 <= metrics["train_accuracy"] <= 1
    assert metrics["train_loss"] >= 0

    assert isinstance(local_model, SimpleCNN)


def test_global_model_is_not_modified():
    model = SimpleCNN(num_classes=10)
    loader = create_dummy_loader()

    original_parameters = {
        name: parameter.detach().clone()
        for name, parameter in model.named_parameters()
    }

    train_local_model(
        global_model=model,
        train_loader=loader,
        device=torch.device("cpu"),
        local_epochs=1,
    )

    for name, parameter in model.named_parameters():
        assert torch.equal(
            parameter,
            original_parameters[name],
        )


def test_evaluate_model():
    model = SimpleCNN(num_classes=10)
    loader = create_dummy_loader()

    metrics = evaluate_model(
        model=model,
        data_loader=loader,
        device=torch.device("cpu"),
    )

    assert "loss" in metrics
    assert "accuracy" in metrics

    assert metrics["loss"] >= 0
    assert 0 <= metrics["accuracy"] <= 1
