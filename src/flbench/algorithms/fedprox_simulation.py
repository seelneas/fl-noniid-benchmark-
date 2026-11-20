from flbench.algorithms.fedprox import train_fedprox_model


def run_fedprox_simulation(
    server,
    client_loaders,
    test_loader,
    num_rounds=10,
    local_epochs=5,
    learning_rate=0.01,
    momentum=0.0,
    weight_decay=0.0,
    mu=0.01,
):
    if num_rounds < 1:
        raise ValueError("num_rounds must be at least 1.")

    if local_epochs < 1:
        raise ValueError("local_epochs must be at least 1.")

    if mu < 0:
        raise ValueError("mu cannot be negative.")

    if not client_loaders:
        raise ValueError("client_loaders cannot be empty.")

    client_ids = sorted(client_loaders.keys())

    for client_id in client_ids:
        if "train" not in client_loaders[client_id]:
            raise ValueError(
                f"Client {client_id} is missing a training loader."
            )

    for round_number in range(1, num_rounds + 1):
        print(f"\nFedProx Round {round_number}/{num_rounds}")

        client_models = []
        client_sizes = []
        client_train_losses = []

        for client_id in client_ids:
            train_loader = client_loaders[client_id]["train"]

            local_model, train_metrics = train_fedprox_model(
                global_model=server.global_model,
                train_loader=train_loader,
                device=server.device,
                local_epochs=local_epochs,
                learning_rate=learning_rate,
                momentum=momentum,
                weight_decay=weight_decay,
                mu=mu,
            )

            client_models.append(local_model)
            client_sizes.append(len(train_loader.dataset))
            client_train_losses.append(train_metrics["train_loss"])

        server.aggregate_updates(client_models, client_sizes)

        test_metrics = server.evaluate_global_model(test_loader)

        total_samples = sum(client_sizes)

        average_train_loss = sum(
            loss * size
            for loss, size in zip(client_train_losses, client_sizes)
        ) / total_samples

        round_metrics = {
            "train_loss": average_train_loss,
            "test_loss": test_metrics["loss"],
            "test_accuracy": test_metrics["accuracy"],
            "num_clients": len(client_ids),
            "total_client_samples": total_samples,
            "mu": mu,
        }

        server.record_round(round_number, round_metrics)

        print(
            f"Train Loss: {average_train_loss:.4f} | "
            f"Test Loss: {test_metrics['loss']:.4f} | "
            f"Test Accuracy: {test_metrics['accuracy']:.4f}"
        )

    return server.round_metrics