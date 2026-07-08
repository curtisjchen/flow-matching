import torch
from models.unet import UNet
from generate import generate
from data import get_dataloader
from torchmetrics.image.fid import FrechetInceptionDistance
from solver import euler_solve
import argparse
import yaml
from pathlib import Path


def eval(config_path="configs/unet_mnist_large.yaml", checkpoint_path="checkpoints/unet_mnist_large_epoch_20.pt", step_counts=[1,25,50,150,250], batchsize=256, samples=10000):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    print("Config:")
    print(yaml.dump(config, default_flow_style=False))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNet(time_in=config["model"]["time_in"],
                 time_out=config["model"]["time_out"],
                 down_in_1=config["model"]["down_in_1"],
                 down_in_2=config["model"]["down_in_2"],
                 down_out_1=config["model"]["down_out_1"],
                 down_out_2=config["model"]["down_out_2"],
                 prefinal=config["model"]["prefinal"])
    
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()
    dataloader = get_dataloader(batch_size=batchsize, train=False)
    c, w, h = dataloader.dataset[0][0].shape
    res_map = {}

    fid = FrechetInceptionDistance(feature=2048, normalize=True, input_img_size=(3, w, h), reset_real_features=False)
    fid = fid.to(device)
    for real_images, _ in dataloader:
        real_images = real_images.to(device)
        real_images = denormalize(real_images)
        real_images = real_images.expand(-1, 3, -1, -1)
        fid.update(real_images, real=True)
    print(f"Real features count: {fid.real_features_num_samples}")
    for steps in step_counts:
        for _ in range(samples // batchsize):
            sample = euler_solve(model=model, N=steps, shape=(batchsize, c, w, h))
            sample = denormalize(sample)
            sample = sample.expand(-1, 3, -1, -1)
            fid.update(sample, real=False)
        res_map[steps] = fid.compute()
        fid.reset()
        print(f"Real features count: {fid.real_features_num_samples}")

    return res_map

def denormalize(images):
    return images * 0.3081 + 0.1307

if __name__ == "__main__":
    res_map = eval()
    for key in res_map:
        print(f"The FID for {key} steps is: {res_map[key]}")
