
import copy

import torch
import torch.nn as nn


def train_fedprox_model(
    global_model,
    train_loader,
    device,
    local_epochs=5,
    learning_rate=0.01,
    momentum=0.0,
    weight_decay=0.0,
    mu=0.01,
):
    """
    Train a local model using the FedProx objective.

    The proximal term discourages local parameters from
    moving too far from the initial global parameters.
    """

    if local_epochs < 1:
        raise ValueError("local_epochs must be at least 1")

    if mu < 0:
        raise ValueError("mu must be non-negative")

    local_model = copy.deepcopy(global_model).to(device)

    # Preserve the initial global parameters as references.
    global_parameters = [
        parameter.detach().clone()
        for parameter in global_model.parameters()
    ]

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.SGD(
        local_model.parameters(),
        lr=learning_rate,
        momentum=momentum,
        weight_decay=weight_decay,
    )

    local_model.train()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for epoch in range(local_epochs):

        for images, labels in train_loader:

            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            outputs = local_model(images)

            # Standard classification loss.
            classification_loss = criterion(outputs, labels)

            # FedProx proximal regularization.
            proximal_term = torch.zeros(
                (),
                device=device,
            )

            for local_parameter, global_parameter in zip(
                local_model.parameters(),
                global_parameters,
            ):
                proximal_term += torch.sum(
                    (local_parameter - global_parameter) ** 2
                )

            loss = (
                classification_loss
                + (mu / 2.0) * proximal_term
            )

            loss.backward()
            optimizer.step()

            batch_size = labels.size(0)

            # Record classification loss separately.
            total_loss += (
                classification_loss.item() * batch_size
            )

            total_correct += (
                outputs.argmax(dim=1) == labels
            ).sum().item()

            total_samples += batch_size

    if total_samples == 0:
        raise ValueError("Training DataLoader is empty")

    metrics = {
        "train_loss": total_loss / total_samples,
        "train_accuracy": total_correct / total_samples,
    }

    return local_model, metrics
