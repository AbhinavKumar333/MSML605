from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def get_cifar10_loaders(batch_size, data_dir="./data", normalize=True):
    transform_list = [
        transforms.Resize((224, 224)),  # Resize for VGG-16
        transforms.ToTensor()
    ]

    if normalize:
        transform_list.append(
            transforms.Normalize((0.4914, 0.4822, 0.4465),  # CIFAR-10 mean
                                 (0.2023, 0.1994, 0.2010))  # CIFAR-10 std
        )

    transform = transforms.Compose(transform_list)

    train_set = datasets.CIFAR10(data_dir, train=True, download=True, transform=transform)
    test_set = datasets.CIFAR10(data_dir, train=False, download=True, transform=transform)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=batch_size)
    return train_loader, test_loader
