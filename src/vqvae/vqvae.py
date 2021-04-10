# %%
import torch
import torch.nn as nn
import torch.nn.functional as F

class Residual(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.ReLU(True),
            # kernel_size=3, stride=1, padding=1 = keeps same spatial dims
            nn.Conv2d(channels, channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(True),
            nn.Conv2d(channels, channels, 1, bias=False),
            nn.BatchNorm2d(channels),
        )

    def forward(self, x):
        return x + self.block(x)


class Encoder(nn.Module):

    def __init__(self, channels, embedding_dim):
        super().__init__()
        self.encoder = nn.Sequential(
            # kernel_size=4, stride=2, padding=1 = effectively half spatial dims
            nn.Conv2d(in_channels=3, out_channels=channels, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(num_features=channels),  # normalize over all dims except channel dim!
            nn.ReLU(True),
            nn.Conv2d(channels, channels, 4, 2, 1),
            nn.BatchNorm2d(channels),
            Residual(channels),
            Residual(channels),
            nn.Conv2d(channels, embedding_dim, 1)
        )

    def forward(self, x):
        # x - [batch, 3, h, w]
        x = x / 255 - 0.5
        return self.encoder(x) 


class GumbelQuantizer(nn.Module):

    def __init__(self, input_dim, num_latents, embedding_dim):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_latents = num_latents
        self.proj = nn.Conv2d(input_dim, num_latents, 1)
        
        # TODO: Create explicit embeddings or reuse from projection matrix?
        self.embed = nn.Embedding(num_latents, embedding_dim)

    def forward(self, x):
        # x: [b, embedding_dim, h/4, w/4]
        x = self.proj(x) # [b, num_latents, h/4, w/4]
        soft_one_hot = F.gumbel_softmax(x, dim=1, hard=False, tau=1.0) # [b, num_latents, h/4, w/4]

        # Project soft_one_hot onto all embeddings
        z_q = torch.einsum('b n h w, n d -> b d h w', soft_one_hot, self.embed.weight) # [b, embedding_dim, h/4, w/4]

        q_z = F.softmax(x, dim=1)
        kl = torch.sum(q_z * torch.log(q_z * self.num_latents + 1e-10), dim=1).mean() # scalar

        return z_q, kl

class Decoder(nn.Module):

    def __init__(self, embedding_dim, channels, output_channels):
        super().__init__()
        self.decoder = nn.Sequential(
            nn.Conv2d(embedding_dim, channels, 1),
            nn.BatchNorm2d(num_features=channels),
            Residual(channels),
            Residual(channels),
            nn.ReLU(True),
            nn.ConvTranspose2d(channels, channels, 4, 2, 1), # upsample 2x
            nn.BatchNorm2d(channels),
            nn.ReLU(True),
            nn.ConvTranspose2d(channels, output_channels, 4, 2, 1),
        )

    def forward(self, x):
        return self.decoder(x)


##

from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
import torchvision.transforms as transforms

BATCH_SIZE=8

train_dataset = CIFAR10("/workspace/data", train=True, transform=transforms.Compose([transforms.PILToTensor()]))
train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE)

i = iter(train_dataloader)

enc = Encoder(channels=64, embedding_dim=64)
out = enc(next(i)[0])
quantizer = GumbelQuantizer(input_dim=64, num_latents=512, embedding_dim=64)
out, kl = quantizer(out)
dec = Decoder(embedding_dim=64, channels=64, output_channels=3)
out = dec(out)
print(out.shape)
print(kl)

# %%
# %%
