
import pytest
import torch

from flbench.models.cnn import SimpleCNN
from flbench.algorithms.fedavg import fedavg_aggregate


def test_fedavg_equal_weights():
    model1 = SimpleCNN()
    model2 = SimpleCNN()

    # Set all parameters to known values
    with torch.no_grad():
        for parameter in model1.parameters():
            parameter.fill_(2.0)

        for parameter in model2.parameters():
            parameter.fill_(4.0)

    global_model = fedavg_aggregate(
        [model1, model2],
        [100, 100],
    )

    # Equal weighting: (2 + 4) / 2 = 3
    for parameter in global_model.parameters():
        assert torch.allclose(
            parameter,
            torch.full_like(parameter, 3.0),
        )


def test_fedavg_weighted_average():
    model1 = SimpleCNN()
    model2 = SimpleCNN()

    with torch.no_grad():
        for parameter in model1.parameters():
            parameter.fill_(2.0)

        for parameter in model2.parameters():
            parameter.fill_(4.0)

    global_model = fedavg_aggregate(
        [model1, model2],
        [100, 300],
    )

    # Weighted average:
    # (2 * 100 + 4 * 300) / 400 = 3.5
    for parameter in global_model.parameters():
        assert torch.allclose(
            parameter,
            torch.full_like(parameter, 3.5),
        )


def test_fedavg_does_not_modify_client_models():
    model1 = SimpleCNN()
    model2 = SimpleCNN()

    original1 = {
        key: value.clone()
        for key, value in model1.state_dict().items()
    }

    original2 = {
        key: value.clone()
        for key, value in model2.state_dict().items()
    }

    fedavg_aggregate([model1, model2], [100, 200])

    for key, value in model1.state_dict().items():
        assert torch.equal(value, original1[key])

    for key, value in model2.state_dict().items():
        assert torch.equal(value, original2[key])


def test_fedavg_invalid_inputs():
    model = SimpleCNN()

    with pytest.raises(ValueError):
        fedavg_aggregate([], [])

    with pytest.raises(ValueError):
        fedavg_aggregate([model], [0])

    with pytest.raises(ValueError):
        fedavg_aggregate([model], [100, 200])
