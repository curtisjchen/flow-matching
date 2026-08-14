import torch
import torchvision
import os
import argparse
from pathlib import Path

from solver import euler_solve, mean_flow_multistep_sample
from utils import build_model
import math

@torch.inference_mode()
def generate(n_steps, checkpoint_path=None, model=None, config=None, samples=16, labels=None, cfg_scale=1.0, suffix="", compile_model=True): 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # --- 1. Resolve Model and Config ---
    if model is None:
        if checkpoint_path is None:
            raise ValueError("Must provide either a 'model' or a 'checkpoint_path'")
        
        print(f"Loading standalone checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
        config = checkpoint['config']
        model = build_model(config).to(device)
        model.load_state_dict(checkpoint['model_state_dict'])
    elif config is None:
        raise ValueError("If providing a live 'model', you must also provide the live 'config' dictionary.")

    # --- 2. Dynamic Dimensions (Always runs!) ---
    c = config['model']['in_channels']
    h = config['model']['h']
    w = config['model']['w']
    shape = (samples, c, h, w)
    
    model.eval()
    
    if compile_model and model is not None:
        print("Compiling model for inference...")
        model = torch.compile(model, mode="reduce-overhead")

    if labels is None:
        labels = torch.full((samples,), config['model']['num_classes'], dtype=torch.long, device=device)
    else:
        labels = labels.to(device=device, dtype=torch.long)
        if labels.numel() != samples:
            raise ValueError("labels must contain exactly one label per requested sample.")
    
    # --- 3. Generate Samples ---
    loss_type = config["training"].get("loss_type", "flow_matching")
    
    if loss_type == "flow_matching":
        sample = euler_solve(
            model=model, N=n_steps, shape=shape, labels=labels, 
            w_val=cfg_scale, null_class_idx=config['model']['num_classes']
        )
    else:
        sample = mean_flow_multistep_sample(
            model=model, N=n_steps, shape=shape, labels=labels, 
            w_val=cfg_scale, null_class_idx=config['model']['num_classes']
        )

    # --- 4. Denormalize and Save ---
    os.makedirs("sample_images", exist_ok=True)
    
    # Reverses the transforms.Normalize((0.5,), (0.5,)) you used in data.py
    sample = (sample * 0.5 + 0.5).clamp(0.0, 1.0)

    dynamic_nrow = int(math.sqrt(samples))
    grid = torchvision.utils.make_grid(sample, nrow=dynamic_nrow)
    
    # Create a safe filename 
    base_name = Path(checkpoint_path).stem if checkpoint_path else f"{config['model'].get('dataset', 'data')}_{loss_type}"
    file_suffix = f"_{suffix}" if suffix else f"_{n_steps}_steps"
    save_path = f"sample_images/{base_name}{file_suffix}.png"
    
    torchvision.utils.save_image(grid.cpu(), fp=save_path)
    print(f"Saved generated grid to {save_path}")
    
    return sample


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Deleted config_path! The checkpoint handles it all now.
    parser.add_argument("--checkpoint_path", type=str, required=True)
    parser.add_argument("--steps_array", type=int, nargs="+", default=[1, 10, 50]) # Let's test a few step counts!
    parser.add_argument("--samples", type=int, default=16)
    args = parser.parse_args()
    
    for n_steps in args.steps_array:
        generate(
            n_steps=n_steps, 
            checkpoint_path=args.checkpoint_path, 
            samples=args.samples
        )