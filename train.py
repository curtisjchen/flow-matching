from models.unet import UNet
from data import get_dataloader
from flow import flow_matching_loss
import torch
import os
import yaml

def train(config_path="configs/unet_mnist.yaml"):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    data = get_dataloader(batch_size=config["training"]["batch_size"], train=True)
    model = UNet(time_in=config["model"]["time_in"],
                 time_out=config["model"]["time_out"],
                 down_in_1=config["model"]["down_in_1"],
                 down_in_2=config["model"]["down_in_2"],
                 down_out_1=config["model"]["down_out_1"],
                 down_out_2=config["model"]["down_out_2"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["training"]["learning_rate"])
    epoch_loss_list = []
    os.makedirs("./checkpoints", exist_ok=True)
    epochs = config["training"]["epochs"]
    for epoch in range(epochs):
        epoch_loss = 0
        batch = 0
        for images, _ in data:
            optimizer.zero_grad()
            loss = flow_matching_loss(model, images)
            loss.backward()
            optimizer.step()
            batch += 1
            if (batch + 1) % 100 == 0:
                print(f"Epoch {epoch+1}/{epochs} | Batch {batch} | Loss: {loss:.4f}")
            epoch_loss += loss.item()
        avg_epoch_loss = epoch_loss / batch
        epoch_loss_list.append(avg_epoch_loss)
        print(f"Epoch {epoch+1}/{epochs} | Avg Loss: {avg_epoch_loss:.4f}")
        save_checkpoint(epoch, model, optimizer, avg_epoch_loss)
    return epoch_loss_list
            

def save_checkpoint(epoch, model, optimizer, avg_epoch_loss):
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': avg_epoch_loss
    }
    filepath = f"./checkpoints/unet_epoch_{epoch}.pt"
    torch.save(checkpoint, filepath)

if __name__ == "__main__":
    train()
