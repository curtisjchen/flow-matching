import torch
import os
import yaml
import argparse
import time
import json
import math
from pathlib import Path
from datetime import datetime
from torch.utils.data import DataLoader

from torchmetrics.image.fid import FrechetInceptionDistance
from data import get_dataloader
from solver import euler_solve, mean_flow_multistep_sample
from utils import build_model

def denormalize(images, verbose=False):
    raw = images * 0.3081 + 0.1307
    if verbose:
        print(f"pre-clamp range: [{raw.min():.3f}, {raw.max():.3f}]")
    return raw.clamp(0.0, 1.0)

def _setup_fid_real_features(fid, dataloader, device, cache_path="fid_real_cache.pt"):
    t0 = time.time()
    if os.path.exists(cache_path):
        cache = torch.load(cache_path, map_location=device, weights_only=True)
        fid.real_features_sum = cache["real_features_sum"].to(device)
        fid.real_features_cov_sum = cache["real_features_cov_sum"].to(device)
        fid.real_features_num_samples = cache["real_features_num_samples"].to(device)
    else:
        print("Caching real dataset FID features (this only happens once)...")
        for real_images, _ in dataloader:
            real_images = real_images.to(device)
            real_images = denormalize(real_images)
            real_images = real_images.expand(-1, 3, -1, -1)
            fid.update(real_images, real=True)
            
        torch.save({
            "real_features_sum": fid.real_features_sum.cpu(),
            "real_features_cov_sum": fid.real_features_cov_sum.cpu(),
            "real_features_num_samples": fid.real_features_num_samples.cpu(),
        }, cache_path)
        
    real_count = getattr(fid.real_features_num_samples, 'item', lambda: fid.real_features_num_samples)()
    print(f"Real features loaded: {real_count} samples | Time: {time.time() - t0:.1f}s")
    return fid

@torch.inference_mode()
def evaluate(config_path, step_counts, checkpoint_path=None, batchsize=256, samples=1000, cfg_scale=1.0, model=None, suffix=""):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
        
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_classes = config["model"]["num_classes"]
    loss_type = config["training"].get("loss_type", "flow_matching")
    config_stem = Path(config_path).stem

    # 1. Load Model
    if model is None:
        if checkpoint_path is None:
            raise ValueError("Must provide either a 'model' or a 'checkpoint_path'")
        model = build_model(config).to(device)
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
        model.load_state_dict(checkpoint["model_state_dict"])
    
    model.eval()
    
    print("Compiling model for inference...")
    model = torch.compile(model, mode="reduce-overhead")

    num_params = sum(p.numel() for p in model.parameters())
    print(f"Evaluating Model | Parameters: {num_params:,}")

    # 2. Setup FID & Dataset
    temp_loader = get_dataloader(batch_size=batchsize, train=False)
    dataloader = DataLoader(
        temp_loader.dataset, 
        batch_size=batchsize, 
        shuffle=False, 
        num_workers=temp_loader.num_workers
    )

    c, w, h = dataloader.dataset[0][0].shape
    fid = FrechetInceptionDistance(feature=2048, normalize=True, input_img_size=(3, w, h), reset_real_features=False)
    fid = fid.to(device)
    fid = _setup_fid_real_features(fid, dataloader, device)

    # 3. Evaluation Loop
    res_map, gen_time_map, fid_time_map = {}, {}, {}
    num_batches = math.ceil(samples / batchsize)

    for steps in step_counts:
        gen_time, fid_eval_time = 0.0, 0.0
        
        for i in range(num_batches):
            current_batch = min(batchsize, samples - i * batchsize)
            
            if cfg_scale > 1.0:
                gen_labels = torch.randint(0, num_classes, (current_batch,), device=device)
            else:
                gen_labels = torch.full((current_batch,), model.null_class_idx, dtype=torch.long, device=device)

            t_gen = time.time()
            shape = (current_batch, c, w, h)
            if loss_type == "flow_matching":
                sample = euler_solve(
                    model=model, N=steps, shape=shape, labels=gen_labels, 
                    w_val=cfg_scale, null_class_idx=model.null_class_idx
                )
            else:
                sample = mean_flow_multistep_sample(
                    model=model, N=steps, shape=shape, labels=gen_labels, 
                    w_val=cfg_scale, null_class_idx=model.null_class_idx
                )
            gen_time += time.time() - t_gen
            
            t_fid = time.time()
            sample = denormalize(sample).expand(-1, 3, -1, -1)
            fid.update(sample, real=False)
            fid_eval_time += time.time() - t_fid
            
        t_compute = time.time()
        res_map[steps] = fid.compute().item()
        fid_eval_time += time.time() - t_compute
        
        gen_time_map[steps] = gen_time
        fid_time_map[steps] = fid_eval_time
        
        print(f"Steps={steps} (CFG={cfg_scale}) | FID: {res_map[steps]:.3f} | Gen time: {gen_time:.1f}s | FID time: {fid_eval_time:.1f}s")
        fid.reset()
 
    # 4. Save Results
    results = {
        "checkpoint_path": checkpoint_path or "passed_from_memory",
        "config": config,
        "step_counts": step_counts,
        "samples": samples,
        "cfg_scale": cfg_scale,
        "fid": {str(k): float(v) for k, v in res_map.items()},
        "timestamp": datetime.now().isoformat()
    }

    os.makedirs("results", exist_ok=True)
    
    file_suffix = f"_{suffix}" if suffix else ""
    save_name = f"results/{config_stem}_cfg_{cfg_scale}_samples_{samples}{file_suffix}.json"
    
    with open(save_name, "w") as f:
        json.dump(results, f, indent=2)

    return res_map

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, default="configs/unet_mnist_large.yaml")
    parser.add_argument("--checkpoint_path", type=str, required=True)
    parser.add_argument("--samples", type=int, default=1024)
    parser.add_argument("--steps_array", type=int, nargs="+", default=[16, 32])
    parser.add_argument("--cfg_scale", type=float, default=1.0)
    args = parser.parse_args()
    
    evaluate(
        config_path=args.config_path, 
        checkpoint_path=args.checkpoint_path, 
        step_counts=args.steps_array, 
        batchsize=256, 
        samples=args.samples,
        cfg_scale=args.cfg_scale
    )