from torchvision import datasets, transforms

transform = transforms.Compose([transforms.ToTensor()])

def get_datasets(data_dir: str):
    train_dataset = datasets.MNIST(
        root=data_dir,
        train=True,
        download=True,
        transform=transform,
    )

    test_dataset = datasets.MNIST(
        root=data_dir,
        train=False,
        download=True,
        transform=transform,
    )

    print(f"Train Samples: {len(train_dataset)}")
    print(f"Test Samples: {len(test_dataset)}")

    return train_dataset, test_dataset
