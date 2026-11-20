
import torch

from flbench.models.cnn import SimpleCNN


def test_model_output_shape():
    model = SimpleCNN(num_classes=10)

    dummy_input = torch.randn(64, 3, 32, 32)
    output = model(dummy_input)

    assert output.shape == (64, 10)


def test_model_forward_pass():
    model = SimpleCNN(num_classes=10)

    dummy_input = torch.randn(1, 3, 32, 32)
    output = model(dummy_input)

    assert torch.isfinite(output).all()


def test_model_parameter_count():
    model = SimpleCNN(num_classes=10)

    total_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
    )

    assert total_parameters > 0
    print(f"Total model parameters: {total_parameters:,}")
