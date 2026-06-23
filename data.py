from torchvision.datasets import MNIST
from torchvision.transforms import Compose, Normalize
from  torch.utils.data import DataLoader

def get_dataloader(batch_size, train):
    return DataLoader(dataset=MNIST, batch_size=batch_size, shuffle=True, train=train)
