from torchvision.datasets import CIFAR100, CIFAR10, MNIST, FashionMNIST  # noqa: F401
from torch.utils.data import DataLoader
import os


def get_data_loader(
    batch_size: int,
    dataset: str = "CIFAR10",
    dataset_root: str = "data",
    additional_dataset_args: dict = {},
    shuffle: bool = False,
    additional_dataloader_args: dict = {},
):
    dataset_class = eval(dataset)
    dataset = dataset_class(
        os.path.join(os.getcwd(), dataset_root), **additional_dataset_args
    )
    return DataLoader(dataset=dataset, shuffle=shuffle, **additional_dataloader_args)


if __name__ == "__main__":
    dataloader = get_data_loader(64, additional_dataset_args={"download": True})
