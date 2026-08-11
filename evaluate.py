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
    """Reverts normalization from [-1, 1] to [0, 1]."""
    # FIX 1: Use 0.5 mean/std to match your data.py transforms
    raw = images * 0.5 + 0.5
    if verbose:
        print(f"pre-clamp range: [{raw.min():.3f}, {raw.max():.3f}]")
    return raw.clamp(0.0, 1.0)

def _setup_fid_real_features(fid, dataloader, device, cache_path="fid_real_cache.pt"):
    """Caches or loads the real dataset features for FID to avoid recomputing them."""
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
            if real_images.shape[1] == 1:
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
def evaluate(config_path, step_counts, checkpoint_path=None, batchsize=256, samples=1000, cfg_scale=1.0, model=None, suffix="", compile_model=True):
    # Only load YAML if we don't have a checkpoint (e.g., if passing a live model from train.py)
    if config_path:
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
            config_stem = Path(config_path).stem
    else:
        config = None
        config_stem = Path(checkpoint_path).stem if checkpoint_path else "live_model"
        
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. Load Model
    if model is None:
        if checkpoint_path is None:
            raise ValueError("Must provide either a 'model' or a 'checkpoint_path'")
        
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
        # FIX 2: Overwrite YAML config with the time-capsule config from the checkpoint!
        config = checkpoint["config"] 
        
        model = build_model(config).to(device)
        model.load_state_dict(checkpoint["model_state_dict"])
    
    model.eval()

    if compile_model:
        print("Compiling model for inference...")
        model = torch.compile(model, mode="reduce-overhead")

    num_params = sum(p.numel() for p in model.parameters())
    print(f"Evaluating Model | Parameters: {num_params:,}")

    num_classes = config["model"]["num_classes"]
    loss_type = config["training"].get("loss_type", "flow_matching")
    dataset_name = config["model"].get("dataset", "mnist")

    # 2. Setup FID & Dataset
    # FIX 3: Pass dataset_name so we don't accidentally load MNIST for a CIFAR-10 model
    temp_loader = get_dataloader(dataset_name=dataset_name, batch_size=batchsize, train=True)
    dataloader = DataLoader(
        temp_loader.dataset, 
        batch_size=batchsize, 
        shuffle=False, 
        num_workers=temp_loader.num_workers
    )   

    # FIX 4: Correct PyTorch shape unpacking (C, H, W)
    c, h, w = dataloader.dataset[0][0].shape
    
    # Cache path specific to the dataset so MNIST and CIFAR caches don't overwrite each other
    cache_path = f"fid_real_cache_{dataset_name}.pt"
    
    fid = FrechetInceptionDistance(feature=2048, normalize=True, input_img_size=(3, h, w), reset_real_features=False)
    fid = fid.to(device)
    fid = _setup_fid_real_features(fid, dataloader, device, cache_path=cache_path)

    # 3. Warmup Compiled Model & CUDA Kernels (Prevents timing leaks on Step 1)
    if compile_model:
        print("Warming up compiled model graph and CUDA kernels...")
        dummy_shape = (batchsize, c, h, w)
        dummy_labels = torch.zeros(batchsize, dtype=torch.long, device=device)
        solver_fn = euler_solve if loss_type == "flow_matching" else mean_flow_multistep_sample
        _ = solver_fn(
            model=model, N=1, shape=dummy_shape, labels=dummy_labels, 
            w_val=1.0, null_class_idx=model.null_class_idx
        )
        torch.cuda.synchronize()
        print("Warmup complete.")

    # 4. Evaluation Loop
    res_map, gen_time_map, fid_time_map = {}, {}, {}
    num_batches = math.ceil(samples / batchsize)

    for steps in step_counts:
        gen_time, fid_eval_time = 0.0, 0.0
        samples_generated = 0
        solver_fn = euler_solve if loss_type == "flow_matching" else mean_flow_multistep_sample
        
        for i in range(num_batches):
            if cfg_scale > 1.0:
                gen_labels = torch.randint(0, num_classes, (batchsize,), device=device)
            else:
                gen_labels = torch.full((batchsize,), model.null_class_idx, dtype=torch.long, device=device)

            shape = (batchsize, c, h, w)
            
            # --- Measure Generation Time ---
            torch.cuda.synchronize()
            t_gen = time.perf_counter()
            
            sample = solver_fn(
                model=model, N=steps, shape=shape, labels=gen_labels, 
                w_val=cfg_scale, null_class_idx=model.null_class_idx
            )
            
            torch.cuda.synchronize()
            gen_time += time.perf_counter() - t_gen
            
            # --- Slice final batch to target sample size ---
            samples_needed = samples - samples_generated
            if samples_needed < batchsize:
                sample = sample[:samples_needed]
            
            samples_generated += sample.shape[0]

            # --- Measure FID Feature Extraction Time ---
            torch.cuda.synchronize()
            t_fid = time.perf_counter()
            
            sample = denormalize(sample)
            if c == 1:
                sample = sample.expand(-1, 3, -1, -1)
            fid.update(sample, real=False)
            
            torch.cuda.synchronize()
            fid_eval_time += time.perf_counter() - t_fid
        
        # --- Measure Final FID Matrix Computation Time ---
        torch.cuda.synchronize()
        t_compute = time.perf_counter()
        res_map[steps] = fid.compute().item()
        torch.cuda.synchronize()
        fid_eval_time += time.perf_counter() - t_compute
        
        gen_time_map[steps] = gen_time
        fid_time_map[steps] = fid_eval_time
        
        print(f"Steps={steps} (CFG={cfg_scale}) | FID: {res_map[steps]:.3f} | Gen time: {gen_time:.2f}s | FID time: {fid_eval_time:.2f}s")
        fid.reset()
 
    # 5. Save Results
    results = {
        "checkpoint_path": checkpoint_path or "passed_from_memory",
        "config": config,
        "step_counts": step_counts,
        "samples": samples,
        "cfg_scale": cfg_scale,
        "fid": {str(k): float(v) for k, v in res_map.items()},
        "gen_time": {str(k): float(v) for k, v in gen_time_map.items()},
        "fid_time": {str(k): float(v) for k, v in fid_time_map.items()},
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
    # config_path is now optional!
    parser.add_argument("--config_path", type=str, default=None)
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