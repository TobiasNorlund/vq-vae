import torch


def VAELoss(x, x_reconstructed, mu, logvar) -> tuple:
    """
        Loss of a variation
    """
    # TODO: Go deeper into this.
    # It is a normal distribution
    kldivloss = ((mu ** 2 + logvar.exp() - 1 - logvar) / 2).mean()

    return rec_loss, kldivloss, tot_loss


def VQVAELoss():
    pass


if __name__ == "__main__":
    x = torch.randn(20)
    x.view(-1, 4)
