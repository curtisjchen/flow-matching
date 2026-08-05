from torchvision.datasets import MNIST 
from torchvision.transforms import Normalize, Compose, ToTensor
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import os

def get_dataloader(batch_size, train): 
    dataset = MNIST(root="./data", train=train, download=True, transform=Compose([ToTensor(), Normalize((0.1307,), (0.3081,))]))
    is_distributed = "LOCAL_RANK" in os.environ
    
    if is_distributed:
        sampler = DistributedSampler(dataset, shuffle=train)
        shuffle_loader = False 
    else:
        sampler = None
        shuffle_loader = train

    return DataLoader(
        dataset, 
        batch_size=batch_size, 
        sampler=sampler,        
        shuffle=shuffle_loader, 
        num_workers=2,         
        pin_memory=True,       
        prefetch_factor=2,
        persistent_workers=True,
        drop_last=train         
    )

if __name__ == "__main__":
    data = get_dataloader(32, True)
    batch = next(iter(data))
    img, label = batch
    print(img.shape, label.shape)
