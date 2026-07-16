import torch, torchvision
from models.unet import UNet
from solver import euler_solve, one_step_sample
import os
import yaml
import argparse
from pathlib import Path


def generate(config_path="configs/unet_mnist_large.yaml", n_steps=150, checkpoint_path="checkpoints/unet_mnist_large_epoch_20.pt", samples=16):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    config_stem = Path(checkpoint_path).stem
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNet(time_in=config["model"]["time_in"],
                time_out=config["model"]["time_out"],
                down_in_1=config["model"]["down_in_1"],
                down_in_2=config["model"]["down_in_2"],
                down_out_1=config["model"]["down_out_1"],
                down_out_2=config["model"]["down_out_2"],
                prefinal=config["model"]["prefinal"])
    
    os.makedirs("sample_images", exist_ok=True)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device) 
    model.eval()
    if config["training"]["loss"] == "flow_matching":
        sample = euler_solve(model=model, N=n_steps, shape=(samples, 1, 28, 28))
    else:
        sample = one_step_sample(model=model, N=n_steps, shape=(samples, 1, 28, 28))
    sample = sample * 0.3081 + 0.1307
    grid = torchvision.utils.make_grid(sample)
    torchvision.utils.save_image(grid.cpu(), fp=f"sample_images/{config_stem}_{n_steps}_steps.png", )
    return sample

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, default="configs/unet_mnist_large.yaml")
    parser.add_argument("--checkpoint_path", type=str, default="checkpoints/unet_mnist_large_epoch_34.pt")
    args = parser.parse_args()
    for n_steps in [5, 25, 150]:
        generate(config_path=args.config_path, n_steps=n_steps, checkpoint_path=args.checkpoint_path, samples=16)