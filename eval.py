import torch
from models.unet import UNet
from generate import generate
from data import get_dataloader
from torchmetrics.image.fid import FrechetInceptionDistance
from solver import euler_solve
import argparse

def eval(configs="configs/unet_mnist.yaml", checkpoint_path="checkpoints/unet_epoch_29.pt", step_counts=[1,10,25,50,100,150,250], batchsize=32, samples=10000):
    model = UNet()
    model.eval()
    checkpoint = torch.load(checkpoint_path, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    dataloader = get_dataloader(batch_size=batchsize, train=False)
    c, w, h = dataloader.dataset[0][0].shape

    res_map = {}
    fid = FrechetInceptionDistance(feature=2048, normalize=True, input_img_size=(3, w, h), reset_real_features=False)
    for real_images, _ in dataloader:
        real_images = denormalize(real_images)
        real_images = real_images.expand(-1, 3, -1, -1)
        fid.update(real_images, real=True)
    for steps in step_counts:
        for _ in range(samples // batchsize):
            sample = euler_solve(model=model, N=steps, shape=(batchsize, c, w, h))
            sample = denormalize(sample)
            sample = sample.expand(-1, 3, -1, -1)
            fid.update(sample, real=False)
        res_map[steps] = fid.compute()
        fid.reset()

    return res_map

def denormalize(images):
    return images * 0.3081 + 0.1307

if __name__ == "__main__":
    res_map = eval()
    for key in res_map:
        print(f"The FID for {key} steps is: {res_map[key]}")
