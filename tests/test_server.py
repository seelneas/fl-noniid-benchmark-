
import torch

from torch.utils.data import DataLoader, TensorDataset

from flbench.models.cnn import SimpleCNN
from flbench.fl.server import FederatedServer


def test_server_initialization():
    model = SimpleCNN()
    server = FederatedServer(model, torch.device("cpu"))

    assert server.global_model is not None
    assert server.device == torch.device("cpu")


def test_model_distribution():
    model = SimpleCNN()
    server = FederatedServer(model, torch.device("cpu"))

    client_models = server.distribute_model([0, 1, 2])

    assert len(client_models) == 3

    for client_model in client_models.values():
        assert client_model is not server.global_model

    # Client copies should have identical initial parameters.
    for client_model in client_models.values():
        for global_param, client_param in zip(
            server.global_model.parameters(),
            client_model.parameters(),
        ):
            assert torch.equal(global_param, client_param)


def test_server_aggregation():
    model1 = SimpleCNN()
    model2 = SimpleCNN()

    server = FederatedServer(
        SimpleCNN(),
        torch.device("cpu"),
    )

    with torch.no_grad():
        for parameter in model1.parameters():
            parameter.fill_(2.0)

        for parameter in model2.parameters():
            parameter.fill_(4.0)

    server.aggregate_updates(
        [model1, model2],
        [100, 300],
    )

    for parameter in server.global_model.parameters():
        assert torch.allclose(
            parameter,
            torch.full_like(parameter, 3.5),
        )


def test_global_model_evaluation():
    model = SimpleCNN()

    server = FederatedServer(
        model,
        torch.device("cpu"),
    )

    images = torch.randn(16, 3, 32, 32)
    labels = torch.randint(0, 10, (16,))

    loader = DataLoader(
        TensorDataset(images, labels),
        batch_size=8,
    )

    metrics = server.evaluate_global_model(loader)

    assert metrics["loss"] >= 0
    assert 0 <= metrics["accuracy"] <= 1


def test_round_recording():
    server = FederatedServer(
        SimpleCNN(),
        torch.device("cpu"),
    )

    record = server.record_round(
        1,
        {
            "loss": 1.5,
            "accuracy": 0.65,
        },
    )

    assert record["round"] == 1
    assert record["loss"] == 1.5
    assert record["accuracy"] == 0.65
    assert len(server.round_metrics) == 1
