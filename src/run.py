import torch
import torch.nn as nn
from typing import Callable


def run_one_vae_epoch(model, data_loader, optimizer):

    if is_training is True:
        optimizer.zero_grad()

    for batch, labels in data_loader:
        (mu, logvar), recons = model(batch)
        # Now, compute the loss


def train_vae(model, n_epochs: int, train_loader, val_loader, optimizer):
    if use_cuda is True:
        assert torch.cuda.is_available()
        model.cuda()
    for epoch in range(n_epochs):

        epoch_train_loss = run_one_vae_epoch()


def eval(model, test_loader):
    pass
