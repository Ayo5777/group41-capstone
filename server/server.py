import flwr as fl
import csv
import argparse
import time
from datetime import datetime

from metriclogger import (
    initialize_dashboard,
    start_round,
    finish_round,
    update_server_status,
    update_client,
    mark_client_uploaded,
    log_event,
)

ROUND_START_TIME = None


class LoggingFedAvg(fl.server.strategy.FedAvg):
    def __init__(self, log_path, num_clients, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.log_path = log_path
        self.num_clients = num_clients

        self._last_fit_clients = 0
        self._last_fit_failures = 0

    def aggregate_fit(self, server_round, results, failures):
        self._last_fit_clients = len(results)
        self._last_fit_failures = len(failures)

        update_server_status("aggregating")
        log_event(
            "INFO",
            f"Server aggregating round {server_round} fit results "
            f"({len(results)} results, {len(failures)} failures)"
        )

        # Update per-client dashboard cards from returned client metrics
        for _, fit_res in results:
            metrics = fit_res.metrics or {}

            client_id = metrics.get("client_id")
            train_loss_last = metrics.get("train_loss_last")
            local_eval_loss = metrics.get("local_eval_loss")
            local_eval_accuracy = metrics.get("local_eval_accuracy")

            if client_id is not None:
                update_client(
                    client_id=int(client_id),
                    status="complete",
                    progress=100,
                    local_accuracy=float(local_eval_accuracy) if local_eval_accuracy is not None else None,
                    local_loss=float(local_eval_loss) if local_eval_loss is not None else (
                        float(train_loss_last) if train_loss_last is not None else None
                    ),
                )
                mark_client_uploaded(int(client_id))

        for failure in failures:
            log_event("WARN", f"Fit failure in round {server_round}: {failure}")

        return super().aggregate_fit(server_round, results, failures)

    def aggregate_evaluate(self, server_round, results, failures):
        aggregated = super().aggregate_evaluate(server_round, results, failures)
        if aggregated is None:
            log_event("WARN", f"No aggregated evaluation produced for round {server_round}")
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

        finish_round(
            round_num=server_round,
            global_accuracy=accuracy_mean,
            global_loss=loss_mean,
            round_time_s=round_time,
        )

        log_event(
            "INFO",
            f"Round {server_round} complete: "
            f"loss={loss_mean:.4f}, acc={accuracy_mean:.4f}, "
            f"time={round_time:.2f}s"
        )

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

    # Optional dashboard/network display values
    parser.add_argument("--latency_ms", type=int, default=0)
    parser.add_argument("--packet_loss_pct", type=float, default=0.0)
    parser.add_argument("--bandwidth_mbps", type=float, default=0.0)

    return parser.parse_args()


def fit_config(server_round):
    global ROUND_START_TIME
    ROUND_START_TIME = time.time()

    # Tell dashboard a new round has started
    start_round(server_round, updates_expected=int(CURRENT_NUM_CLIENTS))
    update_server_status("waiting")
    log_event("INFO", f"Round {server_round} started; waiting for client updates")

    return {"local_epochs": 1, "lr": 0.01}


CURRENT_NUM_CLIENTS = 1


def main():
    global CURRENT_NUM_CLIENTS

    args = parse_args()
    CURRENT_NUM_CLIENTS = args.num_clients

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

    # Initialize dashboard state
    initialize_dashboard(
        num_clients=args.num_clients,
        latency_ms=args.latency_ms,
        packet_loss_pct=args.packet_loss_pct,
        bandwidth_mbps=args.bandwidth_mbps,
    )
    update_server_status("idle")
    log_event("INFO", f"Server initialized at {server_address}")
    log_event(
        "INFO",
        f"Experiment config: rounds={args.rounds}, clients={args.num_clients}, "
        f"timeout={args.round_timeout}s"
    )

    strategy = LoggingFedAvg(
        log_path=log_path,
        num_clients=args.num_clients,
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

    update_server_status("idle")
    log_event("INFO", "Server shutdown")


if __name__ == "__main__":
    main()