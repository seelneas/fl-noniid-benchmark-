import copy
import torch
import torch.nn as nn


def train_scaffold_model(
    global_model,
    train_loader,
    client_control,
    server_control,
    device,
    local_epochs=5,
    learning_rate=0.01,
    momentum=0.0,
    weight_decay=0.0,
):
    if local_epochs < 1:
        raise ValueError("local_epochs must be at least 1.")

    local_model = copy.deepcopy(global_model).to(device)

    global_parameters = [
        parameter.detach().clone()
        for parameter in local_model.parameters()
    ]

    client_control = [
        control.to(device)
        for control in client_control
    ]

    server_control = [
        control.to(device)
        for control in server_control
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
            loss = criterion(outputs, labels)

            loss.backward()

            with torch.no_grad():
                for parameter, client_c, server_c in zip(
                    local_model.parameters(),
                    client_control,
                    server_control,
                ):
                    parameter.grad.add_(server_c - client_c)

            optimizer.step()

            batch_size = labels.size(0)

            total_loss += loss.item() * batch_size
            total_correct += (
                outputs.argmax(dim=1) == labels
            ).sum().item()

            total_samples += batch_size

    if total_samples == 0:
        raise ValueError("Training loader is empty.")

    metrics = {
        "train_loss": total_loss / total_samples,
        "train_accuracy": total_correct / total_samples,
    }

    return local_model, metrics