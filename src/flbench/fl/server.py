
import copy

from flbench.algorithms.fedavg import fedavg_aggregate
from flbench.fl.client import train_local_model, evaluate_model


class FederatedServer:
    """
    Coordinates federated training using FedAvg.
    """

    def __init__(self, global_model, device):
        self.global_model = global_model.to(device)
        self.device = device
        self.round_metrics = []

    def distribute_model(self, client_ids):
        """
        Create independent global-model copies
        for participating clients.
        """
        return {
            client_id: copy.deepcopy(self.global_model)
            for client_id in client_ids
        }

    def aggregate_updates(self, client_models, client_sizes):
        """
        Aggregate client models and update global model.
        """
        self.global_model = fedavg_aggregate(
            client_models,
            client_sizes,
        ).to(self.device)

        return self.global_model

    def evaluate_global_model(self, data_loader):
        """
        Evaluate the current global model on a supplied dataset.
        """
        return evaluate_model(
            self.global_model,
            data_loader,
            self.device,
    )

    def record_round(self, round_number, metrics):
        """
        Save metrics for a communication round.
        """
        record = {
            "round": round_number,
            **metrics,
        }

        self.round_metrics.append(record)

        return record
