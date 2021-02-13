#%%
import pytorch_lightning as pl
import torch
from torch.nn import functional as F
import torchvision
import torchvision.transforms as transforms
# %%

"""
The encoder consists
of 2 strided convolutional layers with stride 2 and window size 4 × 4, 

followed by two residual
3 × 3 blocks (implemented as ReLU, 3x3 conv, ReLU, 1x1 conv), all having 256 hidden units. 

The
decoder similarly has two residual 3 × 3 blocks, followed by two transposed convolutions with stride
2 and window size 4 × 4.
"""

class ResBlock(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.relu = torch.nn.ReLU()
        self.conv3x3 = torch.nn.Conv2d(256, 256, 3, padding=1)
        self.conv1x1 = torch.nn.Conv2d(256, 256, 1)

    def forward(self, inputs):
        x = self.relu(inputs)
        x = self.conv3x3(x)
        x = self.relu(x)
        x = self.conv1x1(x)
        x = x + inputs
        return x


class Encoder(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(in_channels=3, out_channels=256, kernel_size=4, stride=2, padding=1)
        self.conv2 = torch.nn.Conv2d(in_channels=256, out_channels=256, kernel_size=4, stride=2, padding=1)
        self.block1 = ResBlock()
        self.block2 = ResBlock()

    def forward(self, inputs):
        x = self.conv1(inputs)
        x = self.conv2(x)
        x = self.block1(x)
        x = self.block2(x)
        return x


class Decoder(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.block1 = ResBlock()
        self.block2 = ResBlock()
        self.deconv1 = torch.nn.ConvTranspose2d(in_channels=256, out_channels=256, kernel_size=4, stride=2, padding=1)
        self.deconv2 = torch.nn.ConvTranspose2d(in_channels=256, out_channels=3, kernel_size=4, stride=2, padding=1)

    def forward(self, inputs):
        x = self.block1(inputs)
        x = self.block2(x)
        x = self.deconv1(x)
        x = self.deconv2(x)
        return x


class VQVAE(pl.LightningModule):
    
    def __init__(self):
        super().__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()
        self.embeddings = torch.nn.Embedding(512, 256)


    def forward(self, inputs):
        z_e = self.encoder(inputs)  # (b, c, h, w) c=256
        z_e = z_e.permute((0, 2, 3, 1))  # (b, h, w, c)
        
        sq_dist = ((z_e[:, :, :, None, :] - self.embeddings.weight) ** 2).sum(dim=-1)  # (b, h, w, 512)
        e_index = sq_dist.argmin(dim=-1)  # (b, h, w) 
        e = self.embeddings(e_index)  # (b, h, w, 256)
        
        z_q = z_e + (e - z_e).detach()  # Copy gradient from z_q to z_e  (b, h, w, c)
        
        z_q = z_q.permute((0, 3, 1, 2))  # (b, c, h, w)
        z_e = z_e.permute((0, 3, 1, 2)) # (b, c, h, w)
        e = e.permute((0, 3, 1, 2)) # (b, c, h, w)
        
        x_rec = self.decoder(z_q.detach())

        return x_rec, z_e, e

    def training_step(self, inputs, _):
        x, y = inputs
        x_rec, z_e, e = self(x)

        rec_loss = F.mse_loss(x, x_rec)
        emb_loss = F.mse_loss(z_e.detach(), e)
        com_loss = F.mse_loss(z_e, e.detach())
        
        #self.log("rec_loss", rec_loss)
        #self.log("emb_loss", emb_loss)
        #self.log("com_loss", com_loss)

        beta = 1.0
        loss = rec_loss + emb_loss + beta * com_loss
        return loss

    #def validation_step(self, inputs, _):
    #    return self.train_step(inputs, _)

    def configure_optimizers(self):
        #optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        optimizer = torch.optim.SGD(self.parameters(), 1e-3)
        return optimizer

#%%

"""
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

trainset = torchvision.datasets.CIFAR10(root='../../data', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                          shuffle=True, num_workers=2)
"""

from pl_bolts.datamodules import CIFAR10DataModule
datamodule = CIFAR10DataModule('/workspace/data')


trainer = pl.Trainer(max_epochs=30, progress_bar_refresh_rate=10)
vqvae = VQVAE()
trainer.fit(vqvae, datamodule)



#%%



#%%
