from DataSetPart import train_dataset, test_dataset
from Partition import make_split, NUM_CLIENTS, SEED
from torch.utils.data import Subset, DataLoader

batch_size = 32
test_batch_size = 256

def make_client_loaders(train_dataset,test_dataset,num_clients,seed,batch_size,test_batch_size,):
    train_splits = make_split(len(train_dataset),num_clients,seed)
    test_splits = make_split(len(test_dataset),num_clients,seed)
    train_loaders = {}
    test_loaders = {}

    for client_id in range(num_clients):
        train_sub = Subset(train_dataset, train_splits[client_id])
        test_sub = Subset(test_dataset, test_splits[client_id])

        train_loaders[client_id] = DataLoader( train_sub, batch_size = batch_size, shuffle=True, num_workers=0)
        test_loaders[client_id] = DataLoader(test_sub, batch_size=batch_size, shuffle = False, num_workers=0)


    return train_loaders, test_loaders, train_splits, test_splits

train_loaders, test_loaders, train_splits, test_splits = make_client_loaders(train_dataset=train_dataset, test_dataset=test_dataset, num_clients=NUM_CLIENTS, seed=SEED, batch_size= batch_size, test_batch_size=test_batch_size)