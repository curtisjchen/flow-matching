from data import get_dataloader
from flow import flow_matching_loss
from mean_flow import mean_flow_loss, improved_mean_flow_loss
from utils import build_model
import torch
import os
import yaml
import argparse
from pathlib import Path
import time

from evaluate import evaluate
from generate import generate

def train(config_path="configs/unet_mnist.yaml", resume_from=None, reset_scheduler=False, save_all=True):
    os.makedirs("sample_images", exist_ok=True)
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    print("Config:")
    print(yaml.dump(config, default_flow_style=False))

    device_type = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device_type)
    print(f"Using device: {device}")
    
    config_stem = Path(config_path).stem
    data = get_dataloader(batch_size=config["training"]["batch_size"], train=True)
    loss_type = config["training"].get("loss_type", "flow_matching")
    num_classes = config["model"]["num_classes"]
    
    # 1. Build Model using shared util
    model = build_model(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["training"]["learning_rate"])
    scaler = torch.amp.GradScaler('cuda', enabled=torch.cuda.is_available())
    
    epoch_loss_list = []
    os.makedirs("./checkpoints", exist_ok=True)
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

    if resume_from:
        checkpoint = torch.load(resume_from, map_location=device, weights_only=True)
        if reset_scheduler:
            del checkpoint["scheduler_state_dict"]
            print("scheduler reset")
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch_loss_list = checkpoint.get('epoch_loss_list', [])
        start_epoch = checkpoint['epoch'] + 1
        if 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        else:
            for _ in range(start_epoch):
                scheduler.step()

    model = torch.compile(model)

    for epoch in range(start_epoch if resume_from else 0, epochs):
        start = time.time()
        epoch_loss = 0
        batch = 0
        model.train()

        for images, labels in data:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            raw_velocity_mse = None
            
            with torch.autocast(device_type=device_type, dtype=torch.float16, enabled=torch.cuda.is_available()):
                if loss_type == "mean_flow":
                    loss, raw_velocity_mse, diagnostics = mean_flow_loss(
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
                    loss, raw_velocity_mse, diagnostics = improved_mean_flow_loss(
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
                    loss = flow_matching_loss(
                        model=model, x_1=images, labels=labels,
                        p_uncond=config['model']['p_uncond'], w_min=config['model']['w_min'], w_max=config['model']['w_max']
                    )
                    diagnostics = None
                
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            batch += 1
            
            if (batch + 1) % 100 == 0:
                if raw_velocity_mse is None:
                    print(f"Epoch {epoch+1}/{epochs} | Batch {batch+1} | Loss: {loss:.4f}")
                else:
                    print(f"Epoch {epoch+1}/{epochs} | Batch {batch+1} | Objective: {loss:.4f} | Vel MSE: {raw_velocity_mse:.4f} | Max |dU/dr|: {diagnostics['max_abs_d_dr']:.2f}")
            epoch_loss += (raw_velocity_mse if raw_velocity_mse is not None else loss).item()
            
        avg_epoch_loss = epoch_loss / batch
        epoch_loss_list.append(avg_epoch_loss)
        elapsed = time.time() - start
        
        print(f"Epoch {epoch+1}/{epochs} | LR: {optimizer.param_groups[0]['lr']:.6f}| Loss: {avg_epoch_loss:.5f} | Time: {elapsed//60:.0f}m {elapsed%60:.0f}s")
        
        # Save Checkpoint
        if (epoch + 1) % 5 == 0 and save_all:
            save_checkpoint(epoch, model, optimizer, epoch_loss_list, config_stem, scheduler)
            print(f"Epoch {epoch+1} Checkpoint Saved!")

        scheduler.step() 

        # Generate & Eval
        if (epoch + 1) % 10 == 0:
            print(f"\n--- Running Generation & Eval for Epoch {epoch+1} ---")
            n_steps = 32 if loss_type == "flow_matching" else 1
            
            # Use active model from memory (avoids loading checkpoint again)
            gen_labels = (torch.arange(10 * num_classes) % num_classes).to(device)
            
            generate(
                config_path=config_path, n_steps=n_steps, samples=len(gen_labels),
                labels=gen_labels, model=model, suffix=f"epoch_{epoch+1}"
            )
            
            evaluate(
                config_path=config_path, step_counts=[n_steps], batchsize=256, 
                samples=1024, cfg_scale=1.0, model=model, suffix=f"epoch_{epoch+1}"
            )
            print("--------------------------------------------------\n")

    if not save_all:
        save_checkpoint(epoch, model, optimizer, epoch_loss_list, config_stem, scheduler)
    return epoch_loss_list

def save_checkpoint(epoch, model, optimizer, epoch_loss_list, config_stem, scheduler):
    raw_model = getattr(model, "_orig_mod", model)

    checkpoint = {
        'epoch': epoch, 'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch_loss_list' : epoch_loss_list, 'scheduler_state_dict': scheduler.state_dict()
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