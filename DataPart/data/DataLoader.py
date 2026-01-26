from torch.utils.data import Subset, DataLoader
from DataSetPart import get_datasets
from Partition import make_split

def make_client_loaders(
    client_id: int,
    num_clients: int,
    seed: int,
    batch_size: int,
    test_batch_size: int,
    data_dir: str,
):
    train_dataset, test_dataset = get_datasets(data_dir)

    train_splits = make_split(len(train_dataset), num_clients, seed)
    test_splits  = make_split(len(test_dataset),  num_clients, seed)

    train_sub = Subset(train_dataset, train_splits[client_id])
    test_sub  = Subset(test_dataset,  test_splits[client_id])

    trainloader = DataLoader(train_sub, batch_size=batch_size, shuffle=True, num_workers=0)
    testloader  = DataLoader(test_sub,  batch_size=test_batch_size, shuffle=False, num_workers=0)

    return trainloader, testloader
