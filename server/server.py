import flwr as fl
import csv
import argparse
import time
from datetime import datetime

ROUND_START_TIME = None
class LoggingFedAvg(fl.server.strategy.FedAvg):
    def __init__(self, log_path, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.log_path = log_path

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

        with open(self.log_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([server_round, round_time, loss_mean, accuracy_mean, len(results)])

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
    parser.add_argument("--test_name", default = "test_unspecified")
    parser.add_argument("--num_clients", type=int, default=1)
    parser.add_argument("--rounds", type=int, default=3)
    return parser.parse_args()
def fit_config(server_round):
    global ROUND_START_TIME
    ROUND_START_TIME = time.time()
    return {"local_epochs": 1, "lr": 0.01}



def main():
    args = parse_args()

    time_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = f"{args.test_name}_{time_stamp}.csv"
    server_address = "0.0.0.0:8080"

    print("Server Address: ", server_address)
    print("Logging Data To: ", log_path)

    with open(log_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["round", "round_time", "loss_mean", "accuracy_mean", "num_clients"])


    strategy = LoggingFedAvg(
        log_path = log_path,
        fraction_fit = 1.0,
        fraction_evaluate = 1.0,

        # how many clients must be connected before a round starts
        min_available_clients = args.num_clients,

        # how many clients successfully train and send updates each round
        min_fit_clients = args.num_clients,

        # how many clients successfully evaluate the model
        min_evaluate_clients = args.num_clients,

        on_fit_config_fn = fit_config,
        evaluate_metrics_aggregation_fn = weighted_average
    )

    fl.server.start_server(
        server_address = server_address,
        config = fl.server.ServerConfig(num_rounds=args.rounds),
        strategy = strategy
    )


if __name__ == "__main__":
    main()