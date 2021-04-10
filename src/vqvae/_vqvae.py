# %%
import pytorch_lightning as pl
import torch
from torch.nn import functional as F
import torchvision
import torchvision.transforms as transforms
from torch import nn
import numpy as np
from pl_bolts.datamodules import CIFAR10DataModule
from torch.distributions import Categorical
from torchvision import transforms as transform_lib
from pl_bolts.optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR

"""
class Residual(nn.Module):
    def __init__(self, in_channels, num_hiddens, num_residual_hiddens):
        super(Residual, self).__init__()
        self._block = nn.Sequential(
            nn.ReLU(True),
            nn.Conv2d(in_channels=in_channels,
                      out_channels=num_residual_hiddens,
                      kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU(True),
            nn.Conv2d(in_channels=num_residual_hiddens,
                      out_channels=num_hiddens,
                      kernel_size=1, stride=1, bias=False)
        )
    
    def forward(self, x):
        return x + self._block(x)


class ResidualStack(nn.Module):
    def __init__(self, in_channels, num_hiddens, num_residual_layers, num_residual_hiddens):
        super(ResidualStack, self).__init__()
        self._num_residual_layers = num_residual_layers
        self._layers = nn.ModuleList([Residual(in_channels, num_hiddens, num_residual_hiddens)
                             for _ in range(self._num_residual_layers)])

    def forward(self, x):
        for i in range(self._num_residual_layers):
            x = self._layers[i](x)
        return F.relu(x)


class Encoder(nn.Module):
    def __init__(self, in_channels, num_hiddens, num_residual_layers, num_residual_hiddens):
        super(Encoder, self).__init__()

        self._conv_1 = nn.Conv2d(in_channels=in_channels,
                                 out_channels=num_hiddens//2,
                                 kernel_size=4,
                                 stride=2, padding=1)
        self._conv_2 = nn.Conv2d(in_channels=num_hiddens//2,
                                 out_channels=num_hiddens,
                                 kernel_size=4,
                                 stride=2, padding=1)
        self._conv_3 = nn.Conv2d(in_channels=num_hiddens,
                                 out_channels=num_hiddens,
                                 kernel_size=3,
                                 stride=1, padding=1)
        self._residual_stack = ResidualStack(in_channels=num_hiddens,
                                             num_hiddens=num_hiddens,
                                             num_residual_layers=num_residual_layers,
                                             num_residual_hiddens=num_residual_hiddens)

    def forward(self, inputs):
        x = self._conv_1(inputs / 255.0 - 0.5)
        x = F.relu(x)
        
        x = self._conv_2(x)
        x = F.relu(x)
        
        x = self._conv_3(x)
        return self._residual_stack(x)
"""

class Encoder(nn.Module):
    # def __init__(self, channels, latent_dim, embedding_dim):
    def __init__(self, in_channels, num_hiddens, num_residual_layers, num_residual_hiddens):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, num_hiddens, 4, 2, 1, bias=False),
            nn.BatchNorm2d(num_hiddens),
            nn.ReLU(True),
            nn.Conv2d(num_hiddens, num_hiddens, 4, 2, 1, bias=False),
            nn.BatchNorm2d(num_hiddens),
            Residual(num_hiddens),
            Residual(num_hiddens),
            #nn.Conv2d(num_hiddens, latent_dim * embedding_dim, 1)
        )

    def forward(self, x):
        return self.encoder(x / 255.0 - 0.5)

"""
class Decoder(nn.Module):
    def __init__(self, in_channels, num_hiddens, num_residual_layers, num_residual_hiddens):
        super(Decoder, self).__init__()
        
        self._conv_1 = nn.Conv2d(in_channels=in_channels,
                                 out_channels=num_hiddens,
                                 kernel_size=3, 
                                 stride=1, padding=1)
        
        self._residual_stack = ResidualStack(in_channels=num_hiddens,
                                             num_hiddens=num_hiddens,
                                             num_residual_layers=num_residual_layers,
                                             num_residual_hiddens=num_residual_hiddens)
        
        self._conv_trans_1 = nn.ConvTranspose2d(in_channels=num_hiddens, 
                                                out_channels=num_hiddens//2,
                                                kernel_size=4, 
                                                stride=2, padding=1)
        
        self._conv_trans_2 = nn.ConvTranspose2d(in_channels=num_hiddens//2, 
                                                out_channels=num_hiddens//2,
                                                kernel_size=4, 
                                                stride=2, padding=1)

        self._conv_2 = nn.Conv2d(in_channels=num_hiddens//2,
                                 out_channels=3 * 256,
                                 kernel_size=1)        

    def forward(self, inputs):
        x = self._conv_1(inputs)
        x = F.batch_norm(x)
        
        x = self._residual_stack(x)
        
        x = self._conv_trans_1(x)
        x = F.batch_norm(x)
        x = F.relu(x)

        x = self._conv_trans_2(x) 
        x = F.batch_norm(x)
        x = F.relu(x)
        
        x = self._conv_2(x) # (b, 3*256, h, w)

        x = x.reshape((x.shape[0], 3, 256) + x.shape[-2:]).permute(0, 1, 3, 4, 2)
        return Categorical(logits=x) # (b, 256, 3, h, w)
"""

class Residual(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.ReLU(True),
            nn.Conv2d(channels, channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(True),
            nn.Conv2d(channels, channels, 1, bias=False),
            nn.BatchNorm2d(channels)
        )

    def forward(self, x):
        return x + self.block(x)


class Decoder(nn.Module):
    #def __init__(self, channels, latent_dim, embedding_dim):
    def __init__(self, in_channels, num_hiddens, num_residual_layers, num_residual_hiddens):
        super().__init__()
        self.decoder = nn.Sequential(
            nn.Conv2d(in_channels, num_hiddens, 1, bias=False),
            nn.BatchNorm2d(num_hiddens),
            Residual(num_hiddens),
            Residual(num_hiddens),
            nn.ConvTranspose2d(num_hiddens, num_hiddens, 4, 2, 1, bias=False),
            nn.BatchNorm2d(num_hiddens),
            nn.ReLU(True),
            nn.ConvTranspose2d(num_hiddens, num_hiddens, 4, 2, 1, bias=False),
            nn.BatchNorm2d(num_hiddens),
            nn.ReLU(True),
            nn.Conv2d(num_hiddens, 3 * 256, 1)
        )

    def forward(self, x):
        x = self.decoder(x)
        B, _, H, W = x.size()
        x = x.view(B, 3, 256, H, W).permute(0, 1, 3, 4, 2)
        dist = Categorical(logits=x)
        return dist



class VQVAE(pl.LightningModule):
    
    def __init__(self):
        super().__init__()
        self.encoder = Encoder(
            in_channels=3,
            num_hiddens=256,
            num_residual_layers=2,
            num_residual_hiddens=32,
        )
        self.conv_emb = nn.Conv2d(
            in_channels=256,
            out_channels=32,
            kernel_size=1,
            stride=1)
        self.embeddings = nn.Embedding(128, 32)
        self.embeddings.weight.data.uniform_(-1/128, 1/128)
        self.decoder = Decoder(
            in_channels=32,
            num_hiddens=256,
            num_residual_layers=2,
            num_residual_hiddens=32,
        )

    def forward(self, inputs):
        z_e = self.encoder(inputs)  # (b, c, h, w)
        z_e = self.conv_emb(z_e)
        z_e = z_e.permute((0, 2, 3, 1))  # (b, h, w, c)

        sq_dist = (z_e[:, :, :, None, :] - self.embeddings.weight).pow(2).sum(dim=-1)  # (b, h, w, 512)
        e_index = sq_dist.argmin(dim=-1)  # (b, h, w) 
        e = self.embeddings(e_index)  # (b, h, w, c)

        z_q = z_e + (e - z_e).detach()  # Copy gradient from z_q to z_e  (b, h, w, c)

        z_q = z_q.permute((0, 3, 1, 2))  # (b, c, h, w)
        z_e = z_e.permute((0, 3, 1, 2)) # (b, c, h, w)
        e = e.permute((0, 3, 1, 2)) # (b, c, h, w)

        x_rec = self.decoder(z_q) # (b, 256, 3, h, w)

        return x_rec, z_e, e

    def training_step(self, inputs, batch_idx):
        x, y = inputs
        x_rec, z_e, e = self(x)

        #rec_loss = F.mse_loss(x, x_rec)
        #rec_loss = F.cross_entropy(x_rec, x.type(torch.long))
        rec_loss = - x_rec.log_prob(x.type(torch.long)).mean()
        emb_loss = F.mse_loss(z_e.detach(), e)
        com_loss = F.mse_loss(z_e, e.detach())

        self.log("rec_loss", rec_loss)
        self.log("emb_loss", emb_loss)
        self.log("com_loss", com_loss)

        #if global_step % 100 == 0:
        #    self.logger.experiment.add_image('image', torch.cat([x, x_rec], dim=-1)[0], optimizer_idx)

        beta = 0.25
        loss = rec_loss + emb_loss + beta * com_loss
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=5e-4)
        scheduler = LinearWarmupCosineAnnealingLR(optimizer, warmup_epochs=1, max_epochs=40)
        return [optimizer], [scheduler]

#%%

datamodule = CIFAR10DataModule('/workspace/data', batch_size=128)

transform = transform_lib.Compose([transform_lib.PILToTensor()])
datamodule.train_transforms = transform
datamodule.val_transforms = transform
datamodule.test_transforms = transform

trainer = pl.Trainer(gpus=1, max_epochs=30, progress_bar_refresh_rate=10)
vqvae = VQVAE()

trainer.fit(vqvae, datamodule)
