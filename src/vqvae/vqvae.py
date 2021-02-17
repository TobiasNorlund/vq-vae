# %%
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



"""
class ResBlock(torch.nn.Module):

    def __init__(self, c):
        super().__init__()
        self.relu = torch.nn.ReLU()
        self.conv3x3 = torch.nn.Conv2d(c, c, 3, padding=1)
        self.conv1x1 = torch.nn.Conv2d(c, c, 1)

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
        self.block1 = ResBlock(256)
        self.block2 = ResBlock(256)

    def forward(self, inputs):
        x = self.conv1(inputs)
        x = self.conv2(x)
        x = self.block1(x)
        x = self.block2(x)
        return x


class Decoder(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.block1 = ResBlock(64)
        self.block2 = ResBlock(64)
        self.deconv1 = torch.nn.ConvTranspose2d(in_channels=64, out_channels=256, kernel_size=4, stride=2, padding=1)
        self.deconv2 = torch.nn.ConvTranspose2d(in_channels=256, out_channels=3, kernel_size=4, stride=2, padding=1)

    def forward(self, inputs):
        x = self.block1(inputs)
        x = self.block2(x)
        x = self.deconv1(x)
        x = self.deconv2(x)
        return x
"""

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


class VectorQuantizer(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, commitment_cost):
        super(VectorQuantizer, self).__init__()
        
        self._embedding_dim = embedding_dim
        self._num_embeddings = num_embeddings
        
        self._embedding = nn.Embedding(self._num_embeddings, self._embedding_dim)
        self._embedding.weight.data.uniform_(-1/self._num_embeddings, 1/self._num_embeddings)
        self._commitment_cost = commitment_cost

    def forward(self, inputs):
        # convert inputs from BCHW -> BHWC
        inputs = inputs.permute(0, 2, 3, 1).contiguous()
        input_shape = inputs.shape
        
        # Flatten input
        flat_input = inputs.view(-1, self._embedding_dim)
        
        # Calculate distances
        distances = (torch.sum(flat_input**2, dim=1, keepdim=True) 
                    + torch.sum(self._embedding.weight**2, dim=1)
                    - 2 * torch.matmul(flat_input, self._embedding.weight.t()))
            
        # Encoding
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        encodings = torch.zeros(encoding_indices.shape[0], self._num_embeddings, device=inputs.device)
        encodings.scatter_(1, encoding_indices, 1)
        
        # Quantize and unflatten
        quantized = torch.matmul(encodings, self._embedding.weight).view(input_shape)
        
        # Loss
        e_latent_loss = F.mse_loss(quantized.detach(), inputs)
        q_latent_loss = F.mse_loss(quantized, inputs.detach())
        loss = q_latent_loss + self._commitment_cost * e_latent_loss
        
        quantized = inputs + (quantized - inputs).detach()
        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))
        
        # convert quantized from BHWC -> BCHW
        return loss, quantized.permute(0, 3, 1, 2).contiguous(), perplexity, encodings


class VQVAE2(pl.LightningModule):

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
            out_channels=64,
            kernel_size=1,
            stride=1)
        self.vq = VectorQuantizer(512, 64, 0.25)
        self.decoder = Decoder(
            in_channels=64,
            num_hiddens=128,
            num_residual_layers=2,
            num_residual_hiddens=32,
        )

    def forward(self, inputs):
        z = self.encoder(inputs)
        z = self.conv_emb(z)
        loss, quantized, perplexity, _ = self.vq(z)
        x_recon = self.decoder(quantized)
        return loss, x_recon, perplexity

    def training_step(self, inputs, batch_idx):
        inputs, _ = inputs
        vq_loss, data_recon, perplexity = self(inputs)
        recon_error = F.mse_loss(data_recon, inputs)
        loss = recon_error + vq_loss
        self.log("rec_loss", recon_error)
        self.log("vq_loss", vq_loss)
        if batch_idx % 1000 == 0:
            self.logger.experiment.add_image('image', torch.cat([inputs, data_recon], dim=-1)[0], batch_idx)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3, amsgrad=False)
        #optimizer = torch.optim.SGD(self.parameters(), 1e-3)
        return optimizer



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
            out_channels=64,
            kernel_size=1,
            stride=1)
        self.embeddings = nn.Embedding(512, 64)
        self.embeddings.weight.data.uniform_(-1/512, 1/512)
        #self.vq = VectorQuantizer(512, 64, 0.25)
        self.decoder = Decoder(
            in_channels=64,
            num_hiddens=128,
            num_residual_layers=2,
            num_residual_hiddens=32,
        )


    def forward(self, inputs):
        z_e = self.encoder(inputs)  # (b, c, h, w) c=64
        z_e = self.conv_emb(z_e)
        z_e = z_e.permute((0, 2, 3, 1))  # (b, h, w, c)
        z_e = z_e.contiguous()
        z_e_orig_shape = z_e.shape
        
        c = z_e.shape[-1]
        z_e = z_e.view(-1, c)

        dist = \
            torch.sum(z_e ** 2, dim=1, keepdim=True) \
            + torch.sum(self.embeddings.weight ** 2, dim=1) \
            - 2 * torch.matmul(z_e, self.embeddings.weight.t())

        idx = torch.argmin(dist, dim=1)
        e = self.embeddings(idx)
        e = e.view(z_e_orig_shape)

        z_e = z_e.view(z_e_orig_shape)

        emb_loss = F.mse_loss(e, z_e.detach())
        com_loss = F.mse_loss(e.detach(), z_e)

        z_q = z_e + (e - z_e).detach()
        z_q = z_q.permute((0, 3, 1, 2))

        x_rec = self.decoder(z_q)
        rec_loss = F.mse_loss(inputs, x_rec)

        loss = rec_loss + emb_loss + 0.25 * com_loss

        return x_rec, loss, rec_loss


    def training_step(self, inputs, batch_idx):
        global global_step
        x, y = inputs
        x_rec, loss, rec_loss = self(x)

        self.log("rec_loss", rec_loss)

        if global_step % 100 == 0:
            self.logger.experiment.add_image('image', torch.cat([x, x_rec], dim=-1)[0], global_step)
        global_step += 1
        return loss

    #def validation_step(self, inputs, _):
    #    return self.train_step(inputs, _)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        #optimizer = torch.optim.SGD(self.parameters(), 1e-3)
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

"""
from torchvision import datasets
training_data = datasets.CIFAR10(root="/workspace/data", train=True, download=True,
    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,0.5,0.5), (1.0,1.0,1.0))
    ]))
trainloader = torch.utils.data.DataLoader(training_data, batch_size=32,
    shuffle=True, num_workers=2)

validation_data = datasets.CIFAR10(root="/workspace/data", train=False, download=True,
    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,0.5,0.5), (1.0,1.0,1.0))
    ]))
"""

from pl_bolts.datamodules import CIFAR10DataModule
datamodule = CIFAR10DataModule('/workspace/data', batch_size=32)

global global_step
global_step = 0
trainer = pl.Trainer(gpus=1, max_epochs=30, progress_bar_refresh_rate=10)
vqvae = VQVAE()

trainer.fit(vqvae, datamodule)



#%%



#%%
