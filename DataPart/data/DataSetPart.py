from torchvision import datasets, transforms

DATA_DIR = ""
transform = transforms.Compose([transforms.ToTensor()])

train_dataset = datasets.MNIST(
    root=DATA_DIR,
    train = True,
    download = True,
    transform = transform
)

test_dataset = datasets.MNIST(
    root = DATA_DIR,
    train = False,
    download = True,
    transform = transform
)

print(f"Train Samples: {len(train_dataset)}")
print(f"Test Samples: {len(train_dataset)}")