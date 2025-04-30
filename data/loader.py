from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import torch

def get_cifar10_loaders(batch_size, subset, dataset_size, data_dir="./Data", normalize=True, resize_for_vgg=False):
    transform_list = []

    print("Dataset size - {}".format(dataset_size))

    # Resize if model expects 224x224 inputs (like VGG16)
    if resize_for_vgg:
        transform_list.append(transforms.Resize((224, 224)))

    # Always convert to tensor
    transform_list.append(transforms.ToTensor())

    # Optionally normalize
    if normalize:
        transform_list.append(
            transforms.Normalize(
                mean=(0.4914, 0.4822, 0.4465),
                std=(0.2023, 0.1994, 0.2010)
            )
        )

    transform = transforms.Compose(transform_list)

    # Load datasets
    train_set = datasets.CIFAR10(root=data_dir, train=True, download=True, transform=transform)
    test_set = datasets.CIFAR10(root=data_dir, train=False, download=True, transform=transform)

    # Optionally take subset for faster experiments
    if subset:
        train_set = Subset(train_set, range(dataset_size))  # first 5,000 images
        # test_set = Subset(test_set, range(subest_size))    # first 1,000 images

    # Create loaders
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

    return train_loader, test_loader
