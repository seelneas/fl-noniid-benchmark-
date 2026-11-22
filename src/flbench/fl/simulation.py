
import copy

from flbench.fl.client import train_local_model


def run_federated_simulation(
    server,
    client_loaders,
    validation_loader,
    num_rounds=10,
    local_epochs=5,
    learning_rate=0.01,
    momentum=0.0,
    weight_decay=0.0,
):
    """
    Run a complete FedAvg federated learning simulation.

    Args:
        server: Initialized FederatedServer.
        client_loaders: Dictionary containing each client's
                        training and validation DataLoaders.
        validation_loader: Global test DataLoader.
        num_rounds: Number of communication rounds.
        local_epochs: Number of local epochs per round.
        learning_rate: SGD learning rate.
        momentum: SGD momentum.
        weight_decay: SGD weight decay.

    Returns:
        List of metrics recorded for each communication round.
    """

    if num_rounds < 1:
        raise ValueError("num_rounds must be at least 1")

    if local_epochs < 1:
        raise ValueError("local_epochs must be at least 1")

    if not client_loaders:
        raise ValueError("At least one client is required")

    client_ids = sorted(client_loaders.keys())

    for client_id in client_ids:
        if "train" not in client_loaders[client_id]:
            raise ValueError(
                f"Client {client_id} is missing a training DataLoader"
            )
    best_validation_accuracy = -1.0
    best_model_state = None
    best_round = None

    for round_number in range(1, num_rounds + 1):

        print(f"\nCommunication Round {round_number}/{num_rounds}")
        print("-" * 45)

        client_models = []
        client_sizes = []
        client_train_losses = []

        # Each client independently trains a copy
        # of the current global model.
        for client_id in client_ids:

            train_loader = client_loaders[client_id]["train"]

            local_model, train_metrics = train_local_model(
                global_model=server.global_model,
                train_loader=train_loader,
                device=server.device,
                local_epochs=local_epochs,
                learning_rate=learning_rate,
                momentum=momentum,
                weight_decay=weight_decay,
            )

            client_models.append(local_model)
            client_sizes.append(len(train_loader.dataset))
            client_train_losses.append(
                train_metrics["train_loss"]
            )

        # Aggregate all locally trained models.
        server.aggregate_updates(
            client_models,
            client_sizes,
        )

        # Evaluate the updated global model.
        validation_metrics = server.evaluate_global_model(validation_loader)

        current_accuracy = validation_metrics["accuracy"]

        if current_accuracy > best_validation_accuracy:
            best_validation_accuracy = current_accuracy
            best_model_state = copy.deepcopy(
                server.global_model.state_dict()
            )
            best_round = round_number

                # Calculate sample-weighted training loss.
        total_samples = sum(client_sizes)

        average_train_loss = sum(
            loss * size
            for loss, size in zip(
                client_train_losses,
                client_sizes,
            )
        ) / total_samples

        # Record communication-round metrics.
        round_metrics = {
            "train_loss": average_train_loss,
            "validation_loss": validation_metrics["loss"],
            "validation_accuracy": validation_metrics["accuracy"],
            "num_clients": len(client_ids),
            "total_client_samples": total_samples,
}

        server.record_round(round_number, round_metrics)

        print(
    f"Train Loss: {average_train_loss:.4f} | "
    f"Validation Loss: {validation_metrics['loss']:.4f} | "
    f"Validation Accuracy: {validation_metrics['accuracy']:.4f}"
)

    return {
    "history": server.round_metrics,
    "best_model_state": best_model_state,
    "best_round": best_round,
    "best_validation_accuracy": best_validation_accuracy,
}
