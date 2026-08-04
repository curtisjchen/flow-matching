import torch
import torchvision
import os
import yaml
import argparse
from pathlib import Path

from solver import euler_solve, mean_flow_multistep_sample
from utils import build_model

@torch.inference_mode()
def generate(config_path, n_steps, checkpoint_path=None, samples=16, labels=None, model=None, suffix=""):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
        
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config_stem = Path(config_path).stem

    # 1. Load Model (Use provided model to save VRAM, or load from checkpoint)
    if model is None:
        if checkpoint_path is None:
            raise ValueError("Must provide either a 'model' or a 'checkpoint_path'")
        model = build_model(config).to(device)
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
        model.load_state_dict(checkpoint['model_state_dict'])
    
    model.eval()

    # 2. Setup Labels
    if labels is None:
        labels = torch.full((samples,), model.null_class_idx, dtype=torch.long, device=device)
    else:
        labels = labels.to(device=device, dtype=torch.long)
        if labels.numel() != samples:
            raise ValueError("labels must contain exactly one label per requested sample.")

    c = config["model"].get("c", config["model"].get("in_channels", 1))
    w = config["model"].get("w", config["model"].get("image_size", 28))
    h = config["model"].get("h", config["model"].get("image_size", 28))
    
    # 3. Generate Samples
    loss_type = config["training"].get("loss_type", "flow_matching")
    shape = (samples, c, w, h)
    
    if loss_type == "flow_matching":
        sample = euler_solve(
            model=model, N=n_steps, shape=shape, labels=labels, 
            w_val=3.0, null_class_idx=model.null_class_idx
        )
    else:
        sample = mean_flow_multistep_sample(
            model=model, N=n_steps, shape=shape, labels=labels, 
            w_val=3.0, null_class_idx=model.null_class_idx
        )

    # 4. Denormalize and Save
    os.makedirs("sample_images", exist_ok=True)
    sample = (sample * 0.3081 + 0.1307).clamp(0.0, 1.0)
    
    num_classes = config["model"]["num_classes"]
    grid = torchvision.utils.make_grid(sample, nrow=num_classes)
    
    file_suffix = f"_{suffix}" if suffix else f"_{n_steps}_steps"
    save_path = f"sample_images/{config_stem}{file_suffix}.png"
    
    torchvision.utils.save_image(grid.cpu(), fp=save_path)
    print(f"Saved generated grid to {save_path}")
    
    return sample

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, default="configs/unet_mnist_large.yaml")
    parser.add_argument("--checkpoint_path", type=str, required=True)
    parser.add_argument("--steps_array", type=int, nargs="+", default=[1])
    parser.add_argument("--samples", type=int, default=16)
    args = parser.parse_args()
    
    for n_steps in args.steps_array:
        generate(
            config_path=args.config_path, 
            n_steps=n_steps, 
            checkpoint_path=args.checkpoint_path, 
            samples=args.samples
        )