from pathlib import Path
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
import platform


def get_paths(dataset):
    if platform.node() == "Anton":
        train_data_path = Path("D:/code/data/filtered_datasets/", dataset,
                               "train")
        val_data_path = Path("D:/code/data/filtered_datasets/", dataset, "val")
    else:
        raise NotImplementedError("Paths for dataset not implemented for " +
                                  "this machine")

    return train_data_path, val_data_path


def get_icg_loaders(dataset, batch_size):
    N_CLASSES = 11
    train_data_path, val_data_path = get_paths(dataset)

    train_dataset = ImageFolder(root=train_data_path)
    train_loader = DataLoader(
        dataset=train_dataset, batch_size=batch_size, shuffle=True)

    val_dataset = ImageFolder(root=val_data_path)
    val_loader = DataLoader(
        dataset=val_dataset, batch_size=batch_size, shuffle=False)

    return (train_loader, val_loader, N_CLASSES)