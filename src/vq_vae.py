import torch
import torch.nn as nn


# Implemented based on https://arxiv.org/pdf/1711.00937v2.pdf
class VQVAE(nn.Module):
    def __init__(self, D: int, K: int = 512, prior=None):
        """
            args
            D : int
                dimensionality of latent embedding space
            K : int, default 512 as in paper
                number of latent vectors in discrete latent space
            prior : torch.distribitions type, default None
                if None, it will be the uniform Categorical distribution


        """
        self.D = D
        self.K = K  # Default 512 as in paper

        if prior is None:
            probs = torch.ones(self.K) / self.K
            # As in paper
            self.prior = torch.distributions.Categorical(probs)
        else:
            self.prior = prior

    def create_encoder(self):
        """
            For ImageNet (dim 128x128x3), they compress is down to
            z = 32x32x1, i.e., a reduction of 42.6 bits.
        """
        pass

    def create_decoder(self):
        pass
