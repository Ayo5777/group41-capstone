import numpy as np

def make_split(num_samples: int, num_clients: int, seed: int = 67):
    rng = np.random.default_rng(seed)
    indices = np.arange(num_samples)
    rng.shuffle(indices)
    splits = np.array_split(indices, num_clients)
    return splits
