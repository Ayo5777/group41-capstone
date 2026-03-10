import flwr as fl
import csv
import argparse
import time
from typing import List, Optional, Tuple

from flwr.common import (
    FitRes,
    Parameters,
    Scalar,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy.aggregate import aggregate

ROUND_START_TIME = None


class LoggingFedAvgM(fl.server.strategy.FedAvg):
    def __init__(
        self,
        log_path,
        server_momentum: float = 0.9,
        server_lr: float = 1.0,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.log_path = log_path
        self.server_momentum = server_momentum
        self.server_lr = server_lr

        self._last_fit_clients = 0
        self._last_fit_failures = 0

        # Momentum state
        self.current_parameters: Optional[Parameters] = None
        self.momentum_vector: Optional[List] = None

    def initialize_parameters(self, client_manager):
        params = super().initialize_parameters(client_manager)
        self.current_parameters = params
        return params

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures,
    ):
        self._last_fit_clients = len(results)
        self._last_fit_failures = len(failures)

        if not results:
            return None, {}

        if not self.accept_failures and failures:
            return None, {}

        # Standard FedAvg weighted aggregation of client parameters
        weights_results = [
            (parameters_to_ndarrays(fit_res.parameters), fit_res.num_examples)
            for _, fit_res in results
        ]
        fedavg_ndarrays = aggregate(weights_results)

        # If this is the first round or current parameters are unavailable,
        # just take the FedAvg result directly
        if self.current_parameters is None:
            new_parameters = ndarrays_to_parameters(fedavg_ndarrays)
            self.current_parameters = new_parameters
            return new_parameters, {}

        current_ndarrays = parameters_to_ndarrays(self.current_parameters)

        # Compute server "gradient"/delta = aggregated_model - current_model
        delta = [
            fedavg_layer - current_layer
            for fedavg_layer, current_layer in zip(fedavg_ndarrays, current_ndarrays)
        ]

        # Initialize or update momentum buffer
        if self.momentum_vector is None:
            self.momentum_vector = delta
        else:
            self.momentum_vector = [
                self.server_momentum * v + d
                for v, d in zip(self.momentum_vector, delta)
            ]

        # Apply server update
        updated_ndarrays = [
            current_layer + self.server_lr * v
            for current_layer, v in zip(current_ndarrays, self.momentum_vector)
        ]

        new_parameters = ndarrays_to_parameters(updated_ndarrays)
        self.current_parameters = new_parameters

        return new_parameters, {}

    def aggregate_evaluate(self, server_round, results, failures):
        aggregated = super().aggregate_evaluate(server_round, results, failures)
        if aggregated is None:
            return None

        loss_mean, metrics = aggregated
        accuracy_mean = metrics.get("accuracy_mean", 0.0)

        global ROUND_START_TIME
        round_time = 0.0
        if ROUND_START_TIME is not None:
            round_time = time.time() - ROUND_START_TIME

        eval_clients = len(results)
        eval_failures = len(failures)

        with open(self.log_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                server_round,
                round_time,
                loss_mean,
                accuracy_mean,
                self._last_fit_clients,
                self._last_fit_failures,
                eval_clients,
                eval_failures,
            ])

        return loss_mean, metrics
def weighted_average(metrics):
    total_examples = 0
    weighted_acc_sum = 0.0

    for num_examples, m in metrics:
        total_examples += num_examples
        weighted_acc_sum += num_examples * m.get("accuracy", 0.0)

    if total_examples == 0:
        return {"accuracy_mean": 0.0}

    return {"accuracy_mean": weighted_acc_sum / total_examples}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_name", default="test_unspecified")
    parser.add_argument("--num_clients", type=int, default=1)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--round_timeout", type=float, default=60.0)

    # New args
    parser.add_argument("--server_momentum", type=float, default=0.9)
    parser.add_argument("--server_lr", type=float, default=1.0)

    return parser.parse_args()


def fit_config(server_round):
    global ROUND_START_TIME
    ROUND_START_TIME = time.time()
    return {"local_epochs": 1, "lr": 0.01}


def main():
    args = parse_args()

    log_path = f"{args.test_name}.csv"
    server_address = "0.0.0.0:8080"

    print("Server Address: ", server_address)
    print("Logging Data To: ", log_path)
    print("Round Timeout (s): ", args.round_timeout)

    with open(log_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "round",
            "round_time_s",
            "loss_mean",
            "accuracy_mean",
            "fit_clients",
            "fit_failures",
            "eval_clients",
            "eval_failures",
        ])

    strategy = LoggingFedAvgM(
        log_path=log_path,
        server_momentum=args.server_momentum,
        server_lr=args.server_lr,
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_available_clients=int(0.8 * args.num_clients),
        min_fit_clients=int(0.8 * args.num_clients),
        min_evaluate_clients=int(0.8 * args.num_clients),
        on_fit_config_fn=fit_config,
        evaluate_metrics_aggregation_fn=weighted_average,
    )

    fl.server.start_server(
        server_address=server_address,
        config=fl.server.ServerConfig(
            num_rounds=args.rounds,
            round_timeout=args.round_timeout,
        ),
        strategy=strategy,
    )


if __name__ == "__main__":
    main()