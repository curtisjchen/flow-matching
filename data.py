from torchvision.datasets import MNIST 
from torchvision.transforms import Normalize, Compose, ToTensor
from torch.utils.data import DataLoader 

def get_dataloader(batch_size, train): 
    dataset = MNIST(root="./data", train=train, download=True, transform=Compose([ToTensor(), Normalize((0.1307,), (0.3081,))]))
    return DataLoader(dataset=dataset, batch_size=batch_size, shuffle=train)

if __name__ == "__main__":
    data = get_dataloader(32, True)
    batch = next(iter(data))
    img, label = batch
    print(img.shape, label.shape)
    from torchvision.datasets import MNIST
from torchvision.transforms import Compose, Normalize
from  torch.utils.data import DataLoader

def get_dataloader(batch_size, train):
    return DataLoader(dataset=MNIST, batch_size=batch_size, shuffle=True, train=train)
