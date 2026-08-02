import torch, torchvision
from models.unet import UNet
from models.dit import DiT
from solver import euler_solve, mean_flow_multistep_sample
import os
import yaml
import argparse
from pathlib import Path


def generate(config_path="configs/unet_mnist_large.yaml", n_steps=150, checkpoint_path="checkpoints/unet_mnist_large_epoch_20.pt", samples=16, labels=None):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    config_stem = Path(checkpoint_path).stem
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if config["model"]["type"] == "dit":
        model = DiT(hidden_dim=config["model"]["hidden_dim"],
                    num_heads=config["model"]["num_heads"],
                    num_layers=config["model"]["num_layers"],
                    patch_size=config["model"]["patch_size"],
                    in_channels=config["model"]["in_channels"],
                    image_size=config["model"]["image_size"],
                    num_classes=config["model"]["num_classes"])
    elif config["model"]["type"] == "unet":
        model = UNet(time_in=config["model"]["time_in"],
                    time_out=config["model"]["time_out"],
                    down_in_1=config["model"]["down_in_1"],
                    down_in_2=config["model"]["down_in_2"],
                    down_out_1=config["model"]["down_out_1"],
                    down_out_2=config["model"]["down_out_2"],
                    prefinal=config["model"]["prefinal"],
                    num_classes=config["model"]["num_classes"])
    else:
        print("model config not found")
        return
        
    if labels is None:
        labels = torch.full((samples,), model.null_class_idx, dtype=torch.long, device=device)
    else:
        labels = labels.to(device=device, dtype=torch.long)
        if labels.numel() != samples:
            raise ValueError("labels must contain exactly one label per requested sample.")

    c, w, h = config["model"]["c"], config["model"]["w"], config["model"]["h"]
    os.makedirs("sample_images", exist_ok=True)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device) 
    model.eval()
    if config["training"]["loss_type"] == "flow_matching":
        sample = euler_solve(
            model=model, 
            N=n_steps, 
            shape=(samples, c, w, h), 
            labels=labels, w_val=3.0, 
            null_class_idx=model.null_class_idx)
    else:
        sample = mean_flow_multistep_sample(
            model=model, 
            N=n_steps, 
            shape=(samples, c, w, h), 
            labels=labels, w_val=3.0, 
            null_class_idx=model.null_class_idx)
    sample = sample * 0.3081 + 0.1307
    grid = torchvision.utils.make_grid(sample)
    torchvision.utils.save_image(grid.cpu(), fp=f"sample_images/{config_stem}_{n_steps}_steps.png", )
    return sample

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, default="configs/unet_mnist_large.yaml")
    parser.add_argument("--checkpoint_path", type=str, default="checkpoints/unet_mnist_large_epoch_34.pt")
    parser.add_argument("--steps_array", type=int, nargs="+", default=[1])
    parser.add_argument("--samples", type=int, default=16)
    args = parser.parse_args()
    for n_steps in args.steps_array:
        generate(config_path=args.config_path, n_steps=n_steps, checkpoint_path=args.checkpoint_path, samples=args.samples)
