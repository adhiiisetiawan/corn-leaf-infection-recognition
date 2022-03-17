import string
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder

def dataset(data_path: string, crop_size: int, training_size: float, validation_size: float):
    """Process image into dataset format

    Args:
        data_path (string): Directory of dataset.
        crop_size (int): Crop size for Random Resize Crop.
        training_size (float): Number of training size.
        validation_size (float): Number of validation size.

    Returns:
        Torch dataset: Torch dataset that can be use in dataloader.
    """

    preprocess = transforms.Compose([
        transforms.RandomResizedCrop(crop_size, scale=(0.7, 1)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    dataset = ImageFolder(data_path, transform=preprocess)

    train_size = int(training_size * len(dataset))
    val_size = int(validation_size * len(dataset))
    test_size = len(dataset) - train_size - val_size
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, val_size, test_size])

    return train_dataset, val_dataset, test_dataset


def dataloader(train_dataset, val_dataset, test_dataset, batch_size: int):
    """Data loader. Combines a dataset and a sampler, and provides an iterable over
    the given dataset.

    Args:
        train_dataset (torch dataset): Generate from result of dataset function.
        val_dataset (torch dataset): Generate from result of dataset function.
        test_dataset (torch dataset): Generate from result of dataset function.
        batch_size (int): Value of mini batch size.

    Returns:
        Torch Dataloader: Dataloader format that can be use in next phases.
    """

    trainloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    validationloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
    testloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    return trainloader, validationloader, testloader