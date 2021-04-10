
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
import torchvision.transforms as transforms

BATCH_SIZE=64

train_dataset = CIFAR10("/workspace/data", train=True, transform=transforms.Compose([transforms.PILToTensor()]))
train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE)

model = ... #

optimizer = torch.optimizer.Adam()

for epoch in range(10):
    for batch in train_dataloader:
        pass