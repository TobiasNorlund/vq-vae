import torch
import torch.nn as nn
from typing import Tuple


# Highly inspired by https://github.com/SashaMalysheva/Pytorch-VAE/
# And https://arxiv.org/abs/1606.05908
# Taking the same architecture as
class VAE(nn.Module):
    def __init__(self, n_channels, n_kernels, latent_size):
        super().__init__()
        self.n_channels = n_channels
        self.n_kernels = n_kernels
        self.latent_size = latent_size


    def create_encoder(self, encoder_output_size):
        self.encoder = nn.Sequential(
            self._conv(n_channels, n_kernels // 4),
            self._conv(n_channels, n_kernels // 2),
            self._conv(n_channels, n_kernels),
        )
        # A layer to arrive at the mus
        self.q_mu = nn.Linear(, latent_size)
        self.q_vars = nn.Linear(, latent_size)


    def forward(self, x) -> Tuple[tuple, torch.Tensor]:
        # First, derive a latent representation
        latent = self.encoder(x)
        # Then, from the latent variables' PDF of Q(z), sample the means and the vars
        mus, sigmas = self.Q()




        return (mu, sigma), latent

    def Q(self, mu, var):
        # Q is the latent variables' PDF

        pass

    def _conv(self, n_channels, n_kernels):
        return nn.Sequential(

        )
