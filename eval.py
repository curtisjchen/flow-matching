import torch
from models.unet import UNet
from models.dit import DiT
from data import get_dataloader
from torchmetrics.image.fid import FrechetInceptionDistance
from solver import euler_solve, mean_flow_multistep_sample
import argparse
import yaml
from pathlib import Path
import time
import os
import json
import math
from datetime import datetime

def denormalize(images, verbose=False):
    raw = images * 0.3081 + 0.1307
    if verbose:
        print(f"pre-clamp range: [{raw.min():.3f}, {raw.max():.3f}]")
    return raw.clamp(0.0, 1.0)

def eval(config_path="configs/unet_mnist_large.yaml", checkpoint_path="checkpoints/unet_mnist_large_epoch_100.pt", step_counts=[25,100], batchsize=256, samples=1000, cfg_scale=1.0):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    print("Config:")
    print(yaml.dump(config, default_flow_style=False))
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_classes = config["model"]["num_classes"]
    
    # --- 1. Model Setup ---
    if config["model"]["type"] == "dit":
        model = DiT(hidden_dim=config["model"]["hidden_dim"],
                    num_heads=config["model"]["num_heads"],
                    num_layers=config["model"]["num_layers"],
                    patch_size=config["model"]["patch_size"],
                    in_channels=config["model"]["in_channels"],
                    image_size=config["model"]["image_size"],
                    num_classes=num_classes)
    elif config["model"]["type"] == "unet":
        model = UNet(time_in=config["model"]["time_in"],
                    time_out=config["model"]["time_out"],
                    down_in_1=config["model"]["down_in_1"],
                    down_in_2=config["model"]["down_in_2"],
                    down_out_1=config["model"]["down_out_1"],
                    down_out_2=config["model"]["down_out_2"],
                    prefinal=config["model"]["prefinal"],
                    num_classes=num_classes)
    else:
        print("model config not found")
        return
    
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {num_params:,}")
    model = model.to(device)
    model.eval()

    # --- 2. FID Setup & Caching ---
    dataloader = get_dataloader(batch_size=batchsize, train=False)
    c, w, h = dataloader.dataset[0][0].shape
    fid = FrechetInceptionDistance(feature=2048, normalize=True, input_img_size=(3, w, h), reset_real_features=False)
    fid = fid.to(device)

    t0 = time.time()
    CACHE_PATH = "fid_real_cache.pt"
    if os.path.exists(CACHE_PATH):
        cache = torch.load(CACHE_PATH, map_location=device)
        fid.real_features_sum = cache["real_features_sum"].to(device)
        fid.real_features_cov_sum = cache["real_features_cov_sum"].to(device)
        fid.real_features_num_samples = cache["real_features_num_samples"].to(device)
    else:
        for real_images, _ in dataloader:
            real_images = real_images.to(device)
            real_images = denormalize(real_images)
            real_images = real_images.expand(-1, 3, -1, -1)
            fid.update(real_images, real=True)
        torch.save({
            "real_features_sum": fid.real_features_sum.cpu(),
            "real_features_cov_sum": fid.real_features_cov_sum.cpu(),
            "real_features_num_samples": fid.real_features_num_samples.cpu(),
        }, CACHE_PATH)
    
    # Safely get item if it's a tensor, otherwise just print it
    real_count = fid.real_features_num_samples.item() if hasattr(fid.real_features_num_samples, 'item') else fid.real_features_num_samples
    print(f"Real features count: {real_count} | Time: {time.time() - t0:.1f}s")
    
    # --- 3. Generation & Evaluation Loop ---
    res_map, gen_time_map, fid_update_time_map, fid_compute_time_map = {}, {}, {}, {}
    num_batches = math.ceil(samples / batchsize)
    loss_type = config["training"].get("loss_type", "flow_matching")

    for steps in step_counts:
        gen_time, fid_update_time = 0.0, 0.0
        
        for i in range(num_batches):
            # Exact batch sizing to prevent dropping remainders
            current_batch = batchsize if i < num_batches - 1 else samples - (num_batches - 1) * batchsize
            
            # CFG Label Generation
            if cfg_scale > 1.0:
                gen_labels = torch.randint(0, num_classes, (current_batch,), device=device)
            else:
                gen_labels = torch.full((current_batch,), model.null_class_idx, dtype=torch.long, device=device)

            t_gen = time.time()
            if loss_type == "flow_matching":
                sample = euler_solve(
                    model=model, N=steps, shape=(current_batch, c, w, h), 
                    labels=gen_labels, w_val=cfg_scale, null_class_idx=model.null_class_idx
                )
            else:
                # This safely handles both n=1 (one_step) and n>1 for mean_flow
                sample = mean_flow_multistep_sample(
                    model=model, N=steps, shape=(current_batch, c, w, h), 
                    labels=gen_labels, w_val=cfg_scale, null_class_idx=model.null_class_idx
                )
            gen_time += time.time() - t_gen
            
            sample = denormalize(sample)
            sample = sample.expand(-1, 3, -1, -1)
            
            t_upd = time.time()
            fid.update(sample, real=False)
            fid_update_time += time.time() - t_upd
            
        # Compute metrics for this step count
        t2 = time.time()
        res_map[steps] = fid.compute().item()
        fid_compute_time = time.time() - t2
        gen_time_map[steps] = gen_time
        fid_update_time_map[steps] = fid_update_time
        fid_compute_time_map[steps] = fid_compute_time
        
        print(f"Steps={steps} (CFG={cfg_scale}) | FID: {res_map[steps]:.3f} | Gen time: {gen_time:.1f}s | FID eval time: {fid_update_time + fid_compute_time:.1f}s")
        fid.reset()
 
    # --- 4. Save Results ---
    results = {
        "checkpoint_path": checkpoint_path,
        "config_path": config_path,
        "config": config,
        "n_params": num_params,
        "step_counts": step_counts,
        "samples": samples,
        "cfg_scale": cfg_scale,
        "fid": {str(k): float(v) for k, v in res_map.items()},
        "timestamp": datetime.now().isoformat(),
        "sample_gen_time": {str(k): v for k, v in gen_time_map.items()},
        "fid_update_time": {str(k): v for k, v in fid_update_time_map.items()},
        "fid_compute_time": {str(k): v for k, v in fid_compute_time_map.items()},
    }

    os.makedirs("results", exist_ok=True)
    checkpoint_stem = Path(checkpoint_path).stem
    
    # Append CFG scale to the filename so you don't overwrite your unconditional results!
    save_name = f"results/{checkpoint_stem}_cfg_{cfg_scale}_samples_{samples}.json"
    with open(save_name, "w") as f:
        json.dump(results, f, indent=2)

    return res_map

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, default="configs/unet_mnist_large.yaml")
    parser.add_argument("--checkpoint_path", type=str, default="checkpoints/unet_mnist_large_epoch_34.pt")
    parser.add_argument("--samples", type=int, default=1024)
    parser.add_argument("--steps_array", type=int, nargs="+", default=[16, 32])
    parser.add_argument("--cfg_scale", type=float, default=1.0, help="1.0 for unconditional, >1.0 for CFG")
    args = parser.parse_args()
    
    res_map = eval(
        config_path=args.config_path, 
        checkpoint_path=args.checkpoint_path, 
        step_counts=args.steps_array, 
        batchsize=256, 
        samples=args.samples,
        cfg_scale=args.cfg_scale
    )
    print("\n--- Final Results ---")
    for key in res_map:
        print(f"FID @ {key} steps (CFG={args.cfg_scale}): {res_map[key]:.3f}")