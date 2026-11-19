
from pathlib import Path


from torchvision import datasets, transforms


def download_cifar10(data_dir: str = "data"):
    """
    Download CIFAR-10 training and test datasets.
    """
    data_path = Path(data_dir)
    data_path.mkdir(parents=True, exist_ok=True)

    transform = transforms.ToTensor()

    train_dataset = datasets.CIFAR10(
        root=str(data_path),
        train=True,
        download=True,
        transform=transform,
    )

    test_dataset = datasets.CIFAR10(
        root=str(data_path),
        train=False,
        download=True,
        transform=transform,
    )

    return train_dataset, test_dataset
