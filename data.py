import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import os
import glob

def get_dataloader(batch_size, train, dataset_name="mnist"):
    if dataset_name == "cifar10":
        # CIFAR-10 requires 3-channel normalization
        transform = transforms.Compose([
            transforms.RandomHorizontalFlip(), # Free data augmentation - highly recommended for CIFAR!
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) 
        ])
        dataset = torchvision.datasets.CIFAR10(root='./data', train=train, download=True, transform=transform)
    else: 
        # Fallback to your existing MNIST logic
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
        dataset = torchvision.datasets.MNIST(root='./data', train=train, download=True, transform=transform)

    dataset_name = dataset_name.lower()

    if dataset_name == "cifar10":
        kaggle_search = glob.glob("/kaggle/input/**/cifar-10-batches-py", recursive=True)
        
        if kaggle_search:
            data_root = os.path.dirname(kaggle_search[0]) 
            download_flag = False
            print(f"Found CIFAR-10 instantly on Kaggle at: {data_root}")
        else:
            data_root = "./data"
            download_flag = True
            print(f"Loading CIFAR-10 locally: {data_root}")
            
        dataset = torchvision.datasets.CIFAR10(
            root=data_root, 
            train=train, 
            transform=transform, 
            download=download_flag
        )
        
    elif dataset_name == "mnist":
        if os.path.exists("/kaggle/working"):
            data_root = "/kaggle/working/data"
        else:
            data_root = "./data"
            
        print(f"Loading MNIST from: {data_root}")
        
        dataset = torchvision.datasets.MNIST(
            root=data_root, 
            train=train, 
            transform=transform, 
            download=True
        )
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

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
