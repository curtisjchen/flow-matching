from models.unet import UNet
from models.dit import DiT
from data import get_dataloader
from flow import flow_matching_loss
import torch
import os
import yaml
import argparse
from pathlib import Path
import time
import torchvision
from solver import euler_solve

def train(config_path="configs/unet_mnist.yaml", resume_from=None):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    print("Config:")
    print(yaml.dump(config, default_flow_style=False))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    config_stem = Path(config_path).stem
    data = get_dataloader(batch_size=config["training"]["batch_size"], train=True)
    if config["model"]["type"] == "dit":
        model = DiT(hidden_dim=config["model"]["hidden_dim"],
                    num_heads=config["model"]["num_heads"],
                    num_layers=config["model"]["num_layers"],
                    patch_size=config["model"]["patch_size"],
                    in_channels=config["model"]["in_channels"],
                    image_size=config["model"]["image_size"])
    elif config["model"]["type"] == "unet":
        model = UNet(time_in=config["model"]["time_in"],
                    time_out=config["model"]["time_out"],
                    down_in_1=config["model"]["down_in_1"],
                    down_in_2=config["model"]["down_in_2"],
                    down_out_1=config["model"]["down_out_1"],
                    down_out_2=config["model"]["down_out_2"],
                    prefinal=config["model"]["prefinal"])
    else:
        print("model config not found")
        return
    model = model.to(device)
    model = torch.compile(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["training"]["learning_rate"])
    epoch_loss_list = []
    os.makedirs("./checkpoints", exist_ok=True)
    epochs = config["training"]["epochs"]

    warmup_epochs = config["training"]["warmup_epochs"]
    min_lr = config["training"].get("min_lr", 0)

    warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=0.01, end_factor=1.0, total_iters=warmup_epochs
    )
    cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs - warmup_epochs, eta_min=min_lr
    )
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer, schedulers=[warmup_scheduler, cosine_scheduler], milestones=[warmup_epochs]
    )

    if resume_from:
        checkpoint = torch.load(resume_from, map_location=device, weights_only=True)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch_loss_list = checkpoint.get('epoch_loss_list', [])
        start_epoch = checkpoint['epoch'] + 1
        if 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        else:
            for _ in range(start_epoch):
                scheduler.step()

    for epoch in range(start_epoch if resume_from else 0, epochs):
        start = time.time()
        epoch_loss = 0
        batch = 0
        for images, _ in data:
            images = images.to(device)
            optimizer.zero_grad()
            loss = flow_matching_loss(model, images)
            loss.backward()
            optimizer.step()
            batch += 1
            if (batch + 1) % 100 == 0:
                print(f"Epoch {epoch+1}/{epochs} | Batch {batch+1} | Loss: {loss:.4f}")
            epoch_loss += loss.item()
        avg_epoch_loss = epoch_loss / batch
        epoch_loss_list.append(avg_epoch_loss)
        current_lr = optimizer.param_groups[0]['lr']
        elapsed = time.time() - start
        print(f"Epoch {epoch+1}/{epochs} | LR: {current_lr:.6f}| Avg Loss: {avg_epoch_loss:.5f} | Time Taken: {elapsed//60:.0f}m {elapsed%60:.0f}s")
        save_checkpoint(epoch, model, optimizer, epoch_loss_list, config_stem, scheduler)
        scheduler.step() 
        if (epoch + 1) % 10 == 0:
            model.eval()
            with torch.inference_mode():
                sample = euler_solve(model=model, N=50, shape=(16, 1, 28, 28))
            sample = sample * 0.3081 + 0.1307
            grid = torchvision.utils.make_grid(sample.cpu())
            torchvision.utils.save_image(grid, f"sample_images/{config_stem}_epoch_{epoch+1}.png")
            model.train()
    return epoch_loss_list
            

def save_checkpoint(epoch, model, optimizer, epoch_loss_list, config_stem, scheduler):
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch_loss_list' : epoch_loss_list,
        'scheduler_state_dict': scheduler.state_dict()
    }
    filepath = f"./checkpoints/{config_stem}_epoch_{epoch+1}.pt"
    torch.save(checkpoint, filepath)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume_from", type=str, default=None)
    parser.add_argument("--config_path", type=str, default="configs/unet_mnist.yaml")
    args = parser.parse_args()
    train(resume_from=args.resume_from, config_path=args.config_path)
