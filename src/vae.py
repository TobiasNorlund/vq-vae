import torch
import torch.nn as nn
from typing import Tuple


# Highly inspired by https://github.com/SashaMalysheva/Pytorch-VAE/
class VAE(nn.Module):
    def __init__(self):
        super().__init__()

        self.encoder = self.create_encoder()

    def create_encoder(self):
        pass

    def forward(self, x) -> Tuple[tuple, torch.Tensor]:
        latent = self.encoder(x)
        # get the mus
        mu, sigma = self.q()
        return (mu, sigma), latent

    def q(self):
        pass
