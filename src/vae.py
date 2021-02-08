import torch
import torch.nn as nn

# from torch.autograd import Variable # Not needed
from typing import Tuple, Union, List, Dict
import numpy as np
from torch import Tensor as T

# Note: This code is highly inspired by https://github.com/SashaMalysheva/Pytorch-VAE/
# Thank you Sasha Malysheva for a
# And https://arxiv.org/abs/1606.05908
# Taking the same architecture as


def tupfix(x: Union[Tuple[int, int], int]):
    return (x[0], x[1]) if isinstance(x, tuple) else (x, x)


def Conv2d_output_shape(
    N: int,
    C_in: int,
    H_in: int,
    W_in: int,
    C_out: int,
    kernel_size: Union[Tuple[int, int], int],
    dilation: Union[Tuple[int, int], int] = 1,
    stride: Union[Tuple[int, int], int] = 1,
    padding: Union[Tuple[int, int], int] = 0,
):
    kernel_size_x, kernel_size_y = tupfix(kernel_size)
    dilation_x, dilation_y = tupfix(dilation)
    stride_x, stride_y = tupfix(stride)
    padding_x, padding_y = tupfix(padding)

    H_out = (
        int((H_in + 2 * padding_x - dilation_x * (kernel_size_x - 1) - 1) / stride_x)
        + 1
    )
    W_out = (
        int((W_in + 2 * padding_y - dilation_y * (kernel_size_y - 1) - 1) / stride_y)
        + 1
    )

    return N, C_out, H_out, W_out


def GetConv2dSequentialShape(
    input_shape: Tuple[int, int, int, int],
    li: List[Dict[str, Union[Tuple[int, int], int]]],
) -> List[Tuple[int, int, int, int]]:
    shapes = []
    shapes.append(input_shape)
    prev = input_shape
    for i in li:
        i["H_in"] = prev[2]
        i["W_in"] = prev[3]
        shapes.append(Conv2d_output_shape(**i))
        prev = shapes[-1]
        assert shapes[-1] == tuple(
            nn.Conv2d(
                i["C_in"],
                i["C_out"],
                kernel_size=i["kernel_size"],
                stride=i["stride"],
                padding=i["padding"],
            )(torch.randn(shapes[-2])).shape
        )
    return shapes


class VAE(nn.Module):
    def __init__(
        self,
        n_channels: int,
        n_kernels: Union[Tuple[int, int], int],
        latent_size: int,
        kernel_size: Union[Tuple[int, int], int],
        stride: Union[Tuple[int, int], int],
        padding: Union[Tuple[int, int], int],
        img_shape: Tuple[int, int],
    ):
        super().__init__()
        self.n_channels = n_channels
        self.n_kernels = n_kernels
        self.latent_size = latent_size
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.img_shape = img_shape
        self.create_encoder(n_channels, latent_size)
        self.create_decoder()

    def create_decoder(self):
        self.project = self._linear(self.latent_size, self.q_in, relu=False)
        self.decoder = nn.Sequential(
            self._deconv(self.n_kernels, self.n_kernels // 2),
            self._deconv(self.n_kernels // 2, self.n_kernels // 4),
            self._deconv(self.n_kernels // 4, self.n_channels),
            nn.Sigmoid(),
        )

    def create_encoder(self, n_channels, latent_size):
        self.encoder = nn.Sequential(
            self._conv(self.n_channels, self.n_kernels // 4),
            self._conv(self.n_kernels // 4, self.n_kernels // 2),
            self._conv(self.n_kernels // 2, self.n_kernels),
        )

        seq_dicts = [
            {
                "C_in": self.n_channels,
                "C_out": self.n_kernels // 4,
                "N": 1,
                "kernel_size": self.kernel_size,
                "stride": self.stride,
                "padding": self.padding,
            },
            {
                "C_in": self.n_kernels // 4,
                "C_out": self.n_kernels // 2,
                "N": 1,
                "kernel_size": self.kernel_size,
                "stride": self.stride,
                "padding": self.padding,
            },
            {
                "C_in": self.n_kernels // 2,
                "C_out": self.n_kernels,
                "N": 1,
                "kernel_size": self.kernel_size,
                "stride": self.stride,
                "padding": self.padding,
            },
        ]
        self.output_shapes = GetConv2dSequentialShape(
            input_shape=(1, self.n_channels) + self.img_shape, li=seq_dicts
        )
        self.q_in = int(np.prod(self.output_shapes[-1][1:]))
        # A layer to arrive at the mus and the sigmas

        self.q_mu = nn.Linear(self.q_in, latent_size)
        self.q_vars = nn.Linear(self.q_in, latent_size)

    def forward(self, x) -> Tuple[Tuple[T, T], T]:
        # First, derive a latent representation
        x_hat = self.encoder(x)
        x_hat.shape
        # Then, from the latent variables' PDF of Q(z), sample the means and the vars
        mus, log_sigmas = self.Q(x_hat)
        # Once the mus and the
        z_sampled = self.z(mus, log_sigmas)
        z_sampled.shape
        z_projected = self.project(z_sampled).view(
            -1, self.n_kernels, self.output_shapes[-1][2], self.output_shapes[-1][3]
        )  # Project and reshape back to an image

        output = self.decoder(z_projected)

        return (mus, log_sigmas), output

    def Q(self, x_hat):
        # Q is the latent variables' PDF

        x_hat = x_hat.view(-1, self.q_in)
        mu, log_sigma = self.q_mu(x_hat), self.q_vars(x_hat)
        return mu, log_sigma

    def z(self, mu, log_sigma):
        # Sample from z
        sigma = log_sigma.mul(0.5).exp_()
        # Here we sample from a normal distribution
        #
        ret = torch.randn(mu.shape)
        if torch.cuda.is_available() is True:
            ret = ret.cuda()
        return ret.mul(sigma).add_(mu)

    def _conv(self, n_channels, n_kernels):
        return nn.Sequential(
            nn.Conv2d(
                in_channels=n_channels,
                out_channels=n_kernels,
                kernel_size=self.kernel_size,
                stride=self.stride,
                padding=self.padding,
            ),
            nn.BatchNorm2d(n_kernels),
            nn.ReLU(),
        )

    def _deconv(self, n_channels, n_kernels):
        return nn.Sequential(
            nn.ConvTranspose2d(
                n_channels,
                n_kernels,
                kernel_size=self.kernel_size,
                stride=self.stride,
                padding=self.padding,
            ),
            nn.BatchNorm2d(n_kernels),
            nn.ReLU(),
        )

    def _linear(self, n_in, n_out, relu=False):
        if relu is True:
            return nn.Sequential(nn.Linear(n_in, n_out), nn.ReLU())
        else:
            return nn.Linear(n_in, n_out)

    def kldivloss(self, mu, logsigma):
        return ((mu ** 2 + logsigma.exp() - 1 - logsigma) / 2).mean()

    def recloss(self, x, x_hat):
        try:
            return nn.BCELoss(size_average=False)(x_hat, x)
        except Exception:
            print(x_hat)
            print(x)
            raise Exception("ERROR")

    def loss(self, mu, logsigma, x, x_hat):
        loss = {}
        loss["kldivloss"] = self.kldivloss(mu, logsigma)
        loss["recloss"] = self.recloss(x, x_hat)
        tot_loss = loss["kldivloss"] + loss["recloss"]
        return tot_loss, loss


if __name__ == "__main__":
    vae = VAE(
        n_channels=3,
        n_kernels=128,
        latent_size=128,
        kernel_size=4,
        stride=2,
        padding=1,
        img_shape=(32, 32),
    )
