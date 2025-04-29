from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset  # <-- ADD Subset here

def get_cifar10_loaders(batch_size, data_dir="./data", normalize=True, subset=False):
    transform_list = [transforms.ToTensor()]
    if normalize:
        transform_list.append(transforms.Normalize((0.4914, 0.4822, 0.4465),
                                                    (0.2023, 0.1994, 0.2010)))
    transform = transforms.Compose(transform_list)

    train_set = datasets.CIFAR10(data_dir, train=True, download=True, transform=transform)
    test_set = datasets.CIFAR10(data_dir, train=False, download=True, transform=transform)

    if subset:
        train_set = Subset(train_set, range(5000))  # Use only 5k images
        test_set = Subset(test_set, range(1000))    # Use only 1k images

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=batch_size)
    return train_loader, test_loader
