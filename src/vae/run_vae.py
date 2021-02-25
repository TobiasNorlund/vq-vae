import torch
from run import train_vae
from vae import VAE
from data import train_val_test_loader


def main():
    lr = 1e-2
    vae = VAE(
        n_channels=3,
        n_kernels=128,
        latent_size=128,
        kernel_size=4,
        stride=2,
        padding=1,
        img_shape=(32, 32),
    )

    train_l, val_l, test_l = train_val_test_loader(128)

    opt = torch.optim.Adam(vae.parameters(), lr=lr)
    model = train_vae(vae, 200, train_l, val_l, opt)
    print(model)


if __name__ == "__main__":
    main()
