# %%
import pytorch_lightning as pl
import torch
from torch.nn import functional as F
import torchvision
import torchvision.transforms as transforms
from torch import nn


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
        x = self._conv_1(inputs)
        x = F.relu(x)
        
        x = self._conv_2(x)
        x = F.relu(x)
        
        x = self._conv_3(x)
        return self._residual_stack(x)


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
                                                out_channels=3,
                                                kernel_size=4, 
                                                stride=2, padding=1)

    def forward(self, inputs):
        x = self._conv_1(inputs)
        
        x = self._residual_stack(x)
        
        x = self._conv_trans_1(x)
        x = F.relu(x)
        
        return self._conv_trans_2(x)



class VQVAE(pl.LightningModule):
    
    def __init__(self):
        super().__init__()
        self.encoder = Encoder(
            in_channels=3,
            num_hiddens=128,
            num_residual_layers=2,
            num_residual_hiddens=32,
        )
        self.conv_emb = nn.Conv2d(
            in_channels=128,
            out_channels=128,
            kernel_size=1,
            stride=1)
        self.embeddings = nn.Embedding(512, 128)
        self.embeddings.weight.data.uniform_(-1/512, 1/512)
        self.decoder = Decoder(
            in_channels=128,
            num_hiddens=128,
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

        x_rec = self.decoder(z_q)

        return x_rec, z_e, e

    def training_step(self, inputs, _):
        global global_step
        x, y = inputs
        x_rec, z_e, e = self(x)

        rec_loss = F.mse_loss(x, x_rec)
        emb_loss = F.mse_loss(z_e.detach(), e)
        com_loss = F.mse_loss(z_e, e.detach())

        self.log("rec_loss", rec_loss)
        self.log("emb_loss", emb_loss)
        self.log("com_loss", com_loss)

        if global_step % 100 == 0:
            self.logger.experiment.add_image('image', torch.cat([x, x_rec], dim=-1)[0], global_step)
        global_step += 1

        beta = 0.25
        loss = rec_loss + emb_loss + beta * com_loss
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        #optimizer = torch.optim.SGD(self.parameters(), 1e-3)
        return optimizer

#%%

from pl_bolts.datamodules import CIFAR10DataModule
datamodule = CIFAR10DataModule('/workspace/data', batch_size=32)

global global_step
global_step = 0
trainer = pl.Trainer(gpus=1, max_epochs=30, progress_bar_refresh_rate=10)
vqvae = VQVAE()

trainer.fit(vqvae, datamodule)
