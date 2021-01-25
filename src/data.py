from torchvision.datasets import CIFAR100, CIFAR10, MNIST, FashionMNIST  # noqa: F401
from torch.utils.data import DataLoader, random_split
import os
import torch


def get_data_loader(
    batch_size: int,
    dataset: str = "CIFAR10",
    dataset_root: str = "data",
    additional_dataset_args: dict = {},
    shuffle: bool = False,
    additional_dataloader_args: dict = {},
    start_index: int = None,
    end_index: int = None,
):
    """
        Get a single dataloader for an arbitrary dataset
    """
    dataset_class = eval(dataset)
    dataset = dataset_class(
        os.path.join(os.getcwd(), dataset_root), **additional_dataset_args
    )
    return DataLoader(dataset=dataset, shuffle=shuffle, **additional_dataloader_args)


def train_val_test_loader(
    batch_size,
    dataset: str = "CIFAR10",
    fractions=(0.7, 0.2, 0.1),
    dataset_root: str = "data",
    additional_dataset_args: dict = {},
    shuffle: bool = False,
    additional_dataloader_args: dict = {},
    download: bool = False,
    seed: int = 1234,
):
    # assert sum(fractions) == 1.0, "train val test fractions must sum up to 1. Found {}".format(sum(fractions))
    fractions = [x / sum(fractions) for x in fractions]
    dataset_class = eval(dataset)
    dataset = dataset_class(
        os.path.join(os.getcwd(), dataset_root), **additional_dataset_args
    )
    n_rows = dataset.data.shape[0]
    if len(fractions) <= 2:
        train, test = random_split(
            dataset,
            [n_rows * int(fractions[0]), n_rows - n_rows * int(fractions[0])],
            generator=torch.Generator().manual_seed(seed),
        )
        train_loader = DataLoader(
            dataset=train, shuffle=shuffle, **additional_dataloader_args
        )
        test_loader = DataLoader(
            dataset=test, shuffle=shuffle, **additional_dataloader_args
        )
        return train, test
    else:
        train, val, test = random_split(
            dataset,
            [
                n_rows * int(fractions[0]),
                n_rows * int(fractions[1]),
                n_rows - n_rows * int(fractions[0]) - n_rows * int(fractions[1]),
            ],
            generator=torch.Generator().manual_seed(seed),
        )
        train_loader = DataLoader(
            dataset=train, shuffle=shuffle, **additional_dataloader_args
        )
        val_loader = DataLoader(
            dataset=val, shuffle=shuffle, **additional_dataloader_args
        )
        test_loader = DataLoader(
            dataset=test, shuffle=shuffle, **additional_dataloader_args
        )
        return train, val, test


if __name__ == "__main__":
    dataloader = get_data_loader(64, additional_dataset_args={"download": True})
