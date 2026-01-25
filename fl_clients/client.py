import argparse
import socket
import sys
from typing import Dict, List, Tuple

import flwr as fl
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset

# MNIST default dataset
from torchvision import datasets, transforms


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--client_id", type=int, required=True)
    p.add_argument("--num_clients", type=int, default=15)
    p.add_argument("--server", type=str, default="127.0.0.1:8080", help="IP:PORT")
    p.add_argument("--cpu", action="store_true", help="Force CPU")
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--data_dir", type=str, default="data/")
    return p.parse_args()


def get_device(force_cpu: bool) -> torch.device:
    if force_cpu:
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def create_model() -> nn.Module:
    # Simple MNIST MLP
    return nn.Sequential(
        nn.Flatten(),
        nn.Linear(28 * 28, 128),
        nn.ReLU(),
        nn.Linear(128, 10),
    )


def get_parameters(model: nn.Module) -> List[np.ndarray]:
    return [v.detach().cpu().numpy() for _, v in model.state_dict().items()]


def set_parameters(model: nn.Module, parameters: List[np.ndarray]) -> None:
    state_dict = model.state_dict()
    keys = list(state_dict.keys())

    if len(parameters) != len(keys):
        raise ValueError(f"Parameter length mismatch: got {len(parameters)} expected {len(keys)}")

    new_state = {}
    for k, p in zip(keys, parameters):
        new_state[k] = torch.tensor(p)
    model.load_state_dict(new_state, strict=True)


def make_iid_shard(dataset, client_id: int, num_clients: int, seed: int) -> Subset:
    # Deterministic IID shard
    n = len(dataset)
    rng = np.random.default_rng(seed)
    indices = np.arange(n)
    rng.shuffle(indices)

    # Split into num_clients shards (nearly equal)
    shard_sizes = [n // num_clients] * num_clients
    for i in range(n % num_clients):
        shard_sizes[i] += 1

    start = sum(shard_sizes[:client_id])
    end = start + shard_sizes[client_id]
    shard_idx = indices[start:end].tolist()
    return Subset(dataset, shard_idx)


def load_client_data(client_id: int, num_clients: int, batch_size: int, seed: int, data_dir: str) -> Tuple[DataLoader, DataLoader]:
    transform = transforms.Compose([transforms.ToTensor()])

    trainset = datasets.MNIST(root=data_dir, train=True, download=True, transform=transform)
    testset = datasets.MNIST(root=data_dir, train=False, download=True, transform=transform)

    train_subset = make_iid_shard(trainset, client_id, num_clients, seed)

    trainloader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
    testloader = DataLoader(testset, batch_size=256, shuffle=False)
    return trainloader, testloader


def train_one_round(model: nn.Module, trainloader: DataLoader, device: torch.device, epochs: int, lr: float) -> float:
    model.train()
    criterion = nn.CrossEntropyLoss()
    opt = torch.optim.SGD(model.parameters(), lr=lr)

    last_loss = 0.0
    for _ in range(epochs):
        for x, y in trainloader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            opt.step()
            last_loss = float(loss.item())
    return last_loss


@torch.no_grad()
def evaluate_model(model: nn.Module, testloader: DataLoader, device: torch.device) -> Tuple[float, float]:
    model.eval()
    criterion = nn.CrossEntropyLoss()

    total_loss = 0.0
    correct = 0
    total = 0

    for x, y in testloader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = criterion(logits, y)

        total_loss += float(loss.item()) * x.size(0)
        preds = logits.argmax(dim=1)
        correct += int((preds == y).sum().item())
        total += x.size(0)

    loss_avg = total_loss / max(total, 1)
    acc = correct / max(total, 1)
    return loss_avg, acc


class FlowerClient(fl.client.NumPyClient):
    def __init__(self, model: nn.Module, trainloader: DataLoader, testloader: DataLoader, device: torch.device):
        self.model = model
        self.trainloader = trainloader
        self.testloader = testloader
        self.device = device

    def get_parameters(self, config: Dict[str, str]):
        return get_parameters(self.model)

    def fit(self, parameters, config):
        set_parameters(self.model, parameters)

        # Comes from your server's fit_config()
        local_epochs = int(config.get("local_epochs", 1))
        lr = float(config.get("lr", 0.01))

        last_loss = train_one_round(self.model, self.trainloader, self.device, local_epochs, lr)
        updated = get_parameters(self.model)

        num_examples = len(self.trainloader.dataset)
        metrics = {"train_loss_last": last_loss}
        return updated, num_examples, metrics

    def evaluate(self, parameters, config):
        set_parameters(self.model, parameters)
        loss, acc = evaluate_model(self.model, self.testloader, self.device)
        num_examples = len(self.testloader.dataset)

        # IMPORTANT: your server's weighted_average() expects metrics["accuracy"]
        return loss, num_examples, {"accuracy": acc}


def main() -> int:
    args = parse_args()

    if not (0 <= args.client_id < args.num_clients):
        print(f"[client] Invalid client_id={args.client_id} for num_clients={args.num_clients}", file=sys.stderr)
        return 2

    device = get_device(args.cpu)

    trainloader, testloader = load_client_data(
        client_id=args.client_id,
        num_clients=args.num_clients,
        batch_size=args.batch_size,
        seed=args.seed,
        data_dir=args.data_dir,
    )

    model = create_model().to(device)

    # Required debug prints
    print(f"[client] client_id={args.client_id} num_clients={args.num_clients}")
    print(f"[client] local_train_examples={len(trainloader.dataset)}")
    print(f"[client] device={device.type}")
    print(f"[client] server={args.server}")

    client = FlowerClient(model, trainloader, testloader, device)

    try:
        fl.client.start_numpy_client(server_address=args.server, client=client)
    except (ConnectionRefusedError, socket.gaierror, TimeoutError, OSError) as e:
        print(f"[client] Could not connect to server at {args.server}: {type(e).__name__}: {e}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
