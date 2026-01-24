"""
Step 6: Partition Validation Script

Validates:
- shard sizes are correct
- no overlap between client shards
- full coverage (union size equals dataset size)
- determinism (same seed produces identical splits)
- optional: label distribution per client
"""

from collections import Counter
import numpy as np

from fl.data import load_mnist, make_split


def validate_splits(splits, num_samples: int, num_clients: int):
    # 1) Correct number of clients
    assert len(splits) == num_clients, f"Expected {num_clients} splits, got {len(splits)}"

    # 2) Total coverage check
    total = sum(len(shard) for shard in splits)
    assert total == num_samples, f"Total indices across splits {total} != num_samples {num_samples}"

    # 3) Overlap check (set union size must equal num_samples)
    all_indices = np.concatenate(splits)
    unique_count = len(np.unique(all_indices))
    assert unique_count == num_samples, (
        f"Overlap detected: unique indices {unique_count} != num_samples {num_samples}"
    )

    # 4) Index bounds check
    assert all_indices.min() >= 0 and all_indices.max() < num_samples, (
        f"Index out of bounds: min={all_indices.min()}, max={all_indices.max()}, num_samples={num_samples}"
    )

    print("✅ Split validation passed:")
    print(f"   - num_clients: {num_clients}")
    print(f"   - num_samples: {num_samples}")
    print(f"   - total assigned: {total}")
    print(f"   - unique indices: {unique_count}")


def check_determinism(num_samples: int, num_clients: int, seed: int):
    splits_a = make_split(num_samples, num_clients, seed)
    splits_b = make_split(num_samples, num_clients, seed)

    # Compare each shard exactly
    for i in range(num_clients):
        if not np.array_equal(splits_a[i], splits_b[i]):
            raise AssertionError(f"Non-deterministic split detected at client {i}")

    print("✅ Determinism check passed (same seed gives identical splits).")


def label_distribution_per_client(dataset, splits, num_clients: int, max_clients_to_print: int = 5):
    """
    Optional: show label distribution per client to sanity-check heterogeneity.
    Prints first few clients by default.
    """
    print("\nLabel distribution (first few clients):")
    for client_id in range(min(num_clients, max_clients_to_print)):
        shard_indices = splits[client_id]
        labels = [dataset[i][1] for i in shard_indices[:500]]  # sample up to 500 to keep fast
        counts = Counter(labels)
        print(f"Client {client_id}: {dict(sorted(counts.items()))}")


def main():
    DATA_DIR = "../../data"
    NUM_CLIENTS = 15
    SEED = 67

    train_ds, test_ds = load_mnist(DATA_DIR)
    print(f"Train samples: {len(train_ds)}")
    print(f"Test samples: {len(test_ds)}")

    # --- Train splits validation ---
    train_splits = make_split(len(train_ds), NUM_CLIENTS, SEED)
    for cid, shard in enumerate(train_splits):
        print(f"Client {cid}: {len(shard)} train samples")

    validate_splits(train_splits, num_samples=len(train_ds), num_clients=NUM_CLIENTS)
    check_determinism(num_samples=len(train_ds), num_clients=NUM_CLIENTS, seed=SEED)

    # --- Test splits validation (per-client test sets) ---
    test_splits = make_split(len(test_ds), NUM_CLIENTS, SEED + 1)
    for cid, shard in enumerate(test_splits):
        print(f"Client {cid}: {len(shard)} test samples")

    validate_splits(test_splits, num_samples=len(test_ds), num_clients=NUM_CLIENTS)
    check_determinism(num_samples=len(test_ds), num_clients=NUM_CLIENTS, seed=SEED + 1)

    # Optional label distribution sanity check
    label_distribution_per_client(train_ds, train_splits, NUM_CLIENTS, max_clients_to_print=5)

    print("\n✅ All Step 6 checks completed successfully.")


if __name__ == "__main__":
    main()