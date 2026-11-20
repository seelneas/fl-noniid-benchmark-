
import copy

import torch
import torch.nn as nn


def train_local_model(
    global_model,
    train_loader,
    device,
    local_epochs=5,
    learning_rate=0.01,
    momentum=0.0,
    weight_decay=0.0,
):
    """
    Train a local copy of the global model using one client's data.

    Returns:
        local_model: The trained local model.
        metrics: Average training loss and accuracy.
    """

    if local_epochs < 1:
        raise ValueError("local_epochs must be at least 1")

    # Create an independent copy of the global model
    local_model = copy.deepcopy(global_model)
    local_model = local_model.to(device)

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

    # Train over the client's local dataset
    for epoch in range(local_epochs):
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            outputs = local_model(images)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            batch_size = labels.size(0)

            total_loss += loss.item() * batch_size
            total_correct += (
                outputs.argmax(dim=1) == labels
            ).sum().item()
            total_samples += batch_size

    if total_samples == 0:
        raise ValueError("The client's training DataLoader is empty")

    metrics = {
        "train_loss": total_loss / total_samples,
        "train_accuracy": total_correct / total_samples,
    }

    return local_model, metrics


def evaluate_model(model, data_loader, device):
    """
    Evaluate a model on a validation or test DataLoader.

    Returns:
        Dictionary containing average loss and accuracy.
    """

    model = model.to(device)
    model.eval()

    criterion = nn.CrossEntropyLoss()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    with torch.no_grad():
        for images, labels in data_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            batch_size = labels.size(0)

            total_loss += loss.item() * batch_size
            total_correct += (
                outputs.argmax(dim=1) == labels
            ).sum().item()
            total_samples += batch_size

    if total_samples == 0:
        raise ValueError("The evaluation DataLoader is empty")

    return {
        "loss": total_loss / total_samples,
        "accuracy": total_correct / total_samples,
    }
