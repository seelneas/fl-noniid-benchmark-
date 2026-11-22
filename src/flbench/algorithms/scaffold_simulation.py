import copy
import torch

from flbench.algorithms.scaffold import train_scaffold_model
from flbench.fl.client import evaluate_model


def run_scaffold_simulation(
    server,
    client_loaders,
    validation_loader,
    num_rounds=10,
    local_epochs=5,
    learning_rate=0.01,
    momentum=0.0,
    weight_decay=0.0,
):
    if num_rounds < 1:
        raise ValueError("num_rounds must be at least 1.")

    if local_epochs < 1:
        raise ValueError("local_epochs must be at least 1.")

    if learning_rate <= 0:
        raise ValueError("learning_rate must be positive.")

    if not client_loaders:
        raise ValueError("client_loaders cannot be empty.")

    client_ids = sorted(client_loaders.keys())

    for client_id in client_ids:
        if "train" not in client_loaders[client_id]:
            raise ValueError(
                f"Client {client_id} is missing a training loader."
            )

    device = server.device

    # Initialize one client control variate per client.
    client_controls = {}

    for client_id in client_ids:
        client_controls[client_id] = [
            torch.zeros_like(parameter, device=device)
            for parameter in server.global_model.parameters()
        ]

    # Initialize the global control variate.
    server_control = [
        torch.zeros_like(parameter, device=device)
        for parameter in server.global_model.parameters()
    ]
    best_validation_accuracy = -1.0
    best_model_state = None
    best_round = None

    for round_number in range(1, num_rounds + 1):
        print(f"\nSCAFFOLD Round {round_number}/{num_rounds}")

        client_models = []
        client_sizes = []
        client_train_losses = []

        old_client_controls = {
            client_id: [
                control.clone()
                for control in client_controls[client_id]
            ]
            for client_id in client_ids
        }

        for client_id in client_ids:
            train_loader = client_loaders[client_id]["train"]

            global_parameters = [
                parameter.detach().clone()
                for parameter in server.global_model.parameters()
            ]

            local_model, train_metrics = train_scaffold_model(
                global_model=server.global_model,
                train_loader=train_loader,
                client_control=client_controls[client_id],
                server_control=server_control,
                device=device,
                local_epochs=local_epochs,
                learning_rate=learning_rate,
                momentum=momentum,
                weight_decay=weight_decay,
            )

            num_steps = len(train_loader) * local_epochs

            if num_steps == 0:
                raise ValueError(
                    f"Client {client_id} has no training batches."
                )

            # Update this client's control variate.
            with torch.no_grad():
                for index, (old_control, global_parameter, local_parameter) in enumerate(
                    zip(
                        client_controls[client_id],
                        global_parameters,
                        local_model.parameters(),
                    )
                ):
                    new_control = (
                        old_control
                        - server_control[index]
                        + (global_parameter - local_parameter)
                        / (num_steps * learning_rate)
                    )

                    client_controls[client_id][index] = new_control.detach().clone()

            client_models.append(local_model)
            client_sizes.append(len(train_loader.dataset))
            client_train_losses.append(train_metrics["train_loss"])

        # Aggregate local model parameters using sample-size weighting.
        server.aggregate_updates(client_models, client_sizes)

        # Update the global control variate.
        with torch.no_grad():
            for index in range(len(server_control)):
                control_difference = sum(
                    (
                        client_controls[client_id][index]
                        - old_client_controls[client_id][index]
                    )
                    for client_id in client_ids
                )

                server_control[index] += (
                    control_difference / len(client_ids)
                )

        # Evaluate the updated global model.
        validation_metrics = evaluate_model(
            server.global_model,
            validation_loader,
            device,
        )

        current_accuracy = validation_metrics["accuracy"]

        if current_accuracy > best_validation_accuracy:
            best_validation_accuracy = current_accuracy
            best_model_state = copy.deepcopy(server.global_model.state_dict())
            best_round = round_number

        total_samples = sum(client_sizes)

        average_train_loss = sum(
            loss * size
            for loss, size in zip(client_train_losses, client_sizes)
        ) / total_samples

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