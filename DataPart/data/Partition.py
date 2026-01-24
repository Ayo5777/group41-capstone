from DataSetPart import train_dataset, test_dataset
from torch.utils.data import Subset
import numpy as np


def make_split(num_samples: int, num_clients: int, seed: int = 67):
    rng = np.random.default_rng(seed)
    indicies = np.arange(num_samples)
    rng.shuffle(indicies)
    splits = np.array_split(indicies, num_clients)

    return splits



NUM_CLIENTS = 15
SEED = 67
n = 0

splits = make_split(num_samples = 60000, num_clients=NUM_CLIENTS, seed = SEED)

for clientID in range(NUM_CLIENTS):
    clientID = n
    client_indices = splits[clientID]
    client_dataset = Subset(train_dataset, client_indices)
    n = n+1

