from data import get_dataloader
from flow import flow_matching_loss
from mean_flow import mean_flow_loss, improved_mean_flow_loss
from utils import build_model
from count_params import count_params
import torch
import os
import yaml
import argparse
from pathlib import Path
import time
from datetime import timedelta

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from evaluate import evaluate
from generate import generate

import logging
logging.getLogger("torch._inductor.utils").setLevel(logging.ERROR)

from unittest.mock import patch

def train(config_path="configs/unet_mnist.yaml", resume_from=None, reset_scheduler=False, save_all=True):
    is_distributed = "LOCAL_RANK" in os.environ

    if is_distributed:
        local_rank = int(os.environ["LOCAL_RANK"])
        device = torch.device(f"cuda:{local_rank}")
        torch.cuda.set_device(device)
        
        dist.init_process_group(
            backend="nccl", 
            timeout=timedelta(minutes=30), 
            device_id=device
        )
        world_size = dist.get_world_size()
    else:
        local_rank = 0
        world_size = 1
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    device_type = 'cuda' if device.type == 'cuda' else 'cpu'

    if local_rank == 0:
        os.makedirs("sample_images", exist_ok=True)
        os.makedirs("./checkpoints", exist_ok=True)
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        print("Config:")
        print(yaml.dump(config, default_flow_style=False))
        print(f"Using device: {device} | Distributed: {is_distributed} (World Size: {world_size})")
    else:
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)

    config_stem = Path(config_path).stem
    if is_distributed:
        if local_rank == 0:
            get_dataloader(batch_size=config["training"]["batch_size"], train=True)
        dist.barrier()

    data = get_dataloader(batch_size=config["training"]["batch_size"], train=True)
    loss_type = config["training"].get("loss_type", "flow_matching")
    num_classes = config["model"]["num_classes"]
    
    model = build_model(config).to(device)
    if local_rank == 0:
        print(count_params(config_path=config_path))
    
    # 2. Automatically scale LR for effective batch size across multiple GPUs
    base_lr = config["training"]["learning_rate"]
    effective_lr = base_lr * (world_size ** 0.5) if is_distributed else base_lr
    if local_rank == 0 and is_distributed:
        print(f"Base LR: {base_lr} -> Scaled Effective LR: {effective_lr:.6f}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=effective_lr)
    
    # T4 GPU: Retaining standard float16 GradScaler
    scaler = torch.amp.GradScaler('cuda', enabled=device.type == 'cuda')
    
    epoch_loss_list = []
    epochs = config["training"]["epochs"]
    warmup_epochs = config["training"]["warmup_epochs"]
    min_lr = config["training"].get("min_lr", 0)
    
    # Setup Scheduler
    if warmup_epochs == 0:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=min_lr)
    else:
        warmup_scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.01, end_factor=1.0, total_iters=warmup_epochs)
        cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs - warmup_epochs, eta_min=min_lr)
        scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer, schedulers=[warmup_scheduler, cosine_scheduler], milestones=[warmup_epochs])

    start_epoch = 0
    if resume_from:
        checkpoint = torch.load(resume_from, map_location=device, weights_only=True)
        if reset_scheduler:
            del checkpoint["scheduler_state_dict"]
            if local_rank == 0: print("Scheduler reset!")
            
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch_loss_list = checkpoint.get('epoch_loss_list', [])
        start_epoch = checkpoint['epoch'] + 1
        
        if 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        else:
            for _ in range(start_epoch):
                scheduler.step()

    compiled_mean_flow = torch.compile(mean_flow_loss)
    compiled_improved_mean_flow = torch.compile(improved_mean_flow_loss)
    compiled_flow_matching = torch.compile(flow_matching_loss)

    if is_distributed:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank)

    for epoch in range(start_epoch, epochs):
        start = time.time()
        local_epoch_loss = 0
        batch = 0
        model.train()

        # Set sampler epoch for proper multi-GPU shuffling
        if is_distributed and hasattr(data, "sampler") and hasattr(data.sampler, "set_epoch"):
            data.sampler.set_epoch(epoch)

        for images, labels in data:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            raw_velocity_mse = None
            
            # T4 GPU: Autocast with FP16
            with torch.autocast(device_type=device_type, dtype=torch.float16, enabled=device.type == 'cuda'):
                if loss_type == "mean_flow":
                    loss, raw_velocity_mse, diagnostics = compiled_mean_flow(
                        model=model, x_1=images, labels=labels,
                        p_rt=config['training'].get('p_rt', 0.5), p_uncond=config['model']['p_uncond'],
                        w_min=config['model']['w_min'], w_max=config['model']['w_max'],
                        adaptive_loss_power=config['training'].get('adaptive_loss_power', 1.0),
                        adaptive_loss_eps=config['training'].get('adaptive_loss_eps', 1e-2),
                        time_sampler=config['training'].get('time_sampler', 'uniform'),
                        logit_normal_mean=config['training'].get('logit_normal_mean', -0.4),
                        logit_normal_std=config['training'].get('logit_normal_std', 1.0),
                        clamp_d_dr=config['training'].get('clamp_d_dr', None),
                    )
                elif loss_type == "improved_mean_flow":
                    loss, raw_velocity_mse, diagnostics = compiled_improved_mean_flow(
                        model=model, x_1=images, labels=labels,
                        p_rt=config['training'].get('p_rt', 0.5), p_uncond=config['model']['p_uncond'],
                        w_min=config['model']['w_min'], w_max=config['model']['w_max'],
                        adaptive_loss_power=config['training'].get('adaptive_loss_power', 1.0),
                        adaptive_loss_eps=config['training'].get('adaptive_loss_eps', 1e-2),
                        time_sampler=config['training'].get('time_sampler', 'uniform'),
                        logit_normal_mean=config['training'].get('logit_normal_mean', -0.4),
                        logit_normal_std=config['training'].get('logit_normal_std', 1.0),
                        clamp_d_dr=config['training'].get('clamp_d_dr', None),
                    )
                else:
                    loss = compiled_flow_matching(
                        model=model, x_1=images, labels=labels,
                        p_uncond=config['model']['p_uncond'], w_min=config['model']['w_min'], w_max=config['model']['w_max']
                    )
                    diagnostics = {}
                
            scaler.scale(loss).backward()  
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            if (batch + 1) % 50 == 0 and local_rank == 0:
                if raw_velocity_mse is None:
                    print(f"Epoch {epoch+1}/{epochs} | Batch {batch+1} | Loss: {loss:.4f}")
                else:
                    d_dr_val = diagnostics['max_abs_d_dr'].item() if torch.is_tensor(diagnostics['max_abs_d_dr']) else diagnostics['max_abs_d_dr']
                    print(f"Epoch {epoch+1}/{epochs} | Batch {batch+1} | Objective: {loss:.4f} | Vel MSE: {raw_velocity_mse:.4f} | Max |dU/dr|: {d_dr_val:.2f}")

            local_epoch_loss += (raw_velocity_mse if raw_velocity_mse is not None else loss).item()
            batch += 1
            
        if is_distributed:
            loss_tensor = torch.tensor([local_epoch_loss, batch], device=device, dtype=torch.float64)
            dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
            avg_epoch_loss = (loss_tensor[0] / loss_tensor[1]).item()
        else:
            avg_epoch_loss = local_epoch_loss / batch

        epoch_loss_list.append(avg_epoch_loss)
        elapsed = time.time() - start

        if local_rank == 0:
            print(f"Epoch {epoch+1}/{epochs} | LR: {optimizer.param_groups[0]['lr']:.6f} | Loss: {avg_epoch_loss:.5f} | Time: {elapsed//60:.0f}m {elapsed%60:.0f}s")
            
            peak_mem = torch.cuda.max_memory_allocated(device) / (1024 ** 3)
            total_mem = torch.cuda.get_device_properties(device).total_memory / (1024 ** 3)
            free_mem = total_mem - peak_mem
            print(f"GPU Memory: Peak {peak_mem:.2f} GB / Total {total_mem:.2f} GB | Free: {free_mem:.2f} GB\n")

        torch.cuda.reset_peak_memory_stats(device)

        if (epoch + 1) % 5 == 0 and save_all and local_rank == 0:
            save_checkpoint(epoch, model, optimizer, epoch_loss_list, config_stem, scheduler)
            print(f"Epoch {epoch+1} Checkpoint Saved!")

        scheduler.step() 

        # Generate & Eval (Rank 0 Only)
        if (epoch + 1) % 10 == 0 and local_rank == 0:
            print(f"\n--- Running Generation & Eval for Epoch {epoch+1} ---")
            n_steps = 32 if loss_type == "flow_matching" else 1
            
            eval_model = model.module if hasattr(model, "module") else model
            eval_model = getattr(eval_model, "_orig_mod", eval_model)
            eval_model.eval()
            
            gen_labels = (torch.arange(10 * num_classes) % num_classes).to(device)

            with patch('torch.distributed.is_initialized', return_value=False):
                generate(
                    config_path=config_path, n_steps=n_steps, samples=len(gen_labels),
                    labels=gen_labels, model=eval_model, suffix=f"epoch_{epoch+1}"
                )
                
                evaluate(
                    config_path=config_path, step_counts=[n_steps], batchsize=256, 
                    samples=8192, cfg_scale=1.0, model=eval_model, suffix=f"epoch_{epoch+1}"
                )
            
            model.train()
            print("--------------------------------------------------\n")

        # Sync processes before starting next epoch
        if is_distributed:
            dist.barrier()

    if not save_all and local_rank == 0:
        save_checkpoint(epochs - 1, model, optimizer, epoch_loss_list, config_stem, scheduler)

    if is_distributed:
        dist.destroy_process_group()

    return epoch_loss_list

def save_checkpoint(epoch, model, optimizer, epoch_loss_list, config_stem, scheduler):
    raw_model = model.module if hasattr(model, "module") else model
    raw_model = getattr(raw_model, "_orig_mod", raw_model)

    checkpoint = {
        'epoch': epoch, 
        'model_state_dict': raw_model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch_loss_list': epoch_loss_list, 
        'scheduler_state_dict': scheduler.state_dict()
    }
    filepath = f"./checkpoints/{config_stem}_epoch_{epoch+1}.pt"
    torch.save(checkpoint, filepath)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume_from", type=str, default=None)
    parser.add_argument("--config_path", type=str, default="configs/unet_mnist.yaml")
    parser.add_argument("--reset_scheduler", action="store_true")
    args = parser.parse_args()
    train(resume_from=args.resume_from, config_path=args.config_path, reset_scheduler=args.reset_scheduler)