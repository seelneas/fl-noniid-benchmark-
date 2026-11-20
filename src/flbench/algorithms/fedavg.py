
import copy
import torch


def fedavg_aggregate(client_models, client_sizes):
    """
    Aggregate locally trained client models using
    sample-size-weighted Federated Averaging.

    Args:
        client_models: List of locally trained PyTorch models.
        client_sizes: Number of training samples for each client.

    Returns:
        A new global model containing the aggregated parameters.
    """

    if len(client_models) == 0:
        raise ValueError("At least one client model is required")

    if len(client_models) != len(client_sizes):
        raise ValueError(
            "Number of client models and client sizes must match"
        )

    if any(size <= 0 for size in client_sizes):
        raise ValueError("Client sizes must be positive")

    total_samples = sum(client_sizes)

    # Create an independent copy of the first client model
    global_model = copy.deepcopy(client_models[0])

    # Obtain the state dictionaries
    global_state = global_model.state_dict()
    client_states = [
        model.state_dict() for model in client_models
    ]

    # Weighted average of model parameters
    for key in global_state:
        global_state[key] = torch.zeros_like(global_state[key])

        for client_state, client_size in zip(
            client_states, client_sizes
        ):
            weight = client_size / total_samples

            global_state[key] += (
                client_state[key] * weight
            )

    global_model.load_state_dict(global_state)

    return global_model
