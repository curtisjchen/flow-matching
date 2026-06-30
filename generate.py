import torch, torchvision
from models.unet import UNet
from solver import euler_solve
import os

def generate(n_steps=150, checkpoint_path="checkpoints/unet_epoch_9.pt", samples=16, output_path="sample_images/"):
    model = UNet()
    os.makedirs("sample_images", exist_ok=True)
    checkpoint = torch.load(checkpoint_path, weights_only=True)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    sample = euler_solve(model=model, N=n_steps, shape=(samples, 1, 28, 28))
    sample = sample * 0.3081 + 0.1307
    grid = torchvision.utils.make_grid(sample)
    torchvision.utils.save_image(grid, output_path)
    return sample

if __name__ == "__main__":
    for n_steps in [1, 5, 25]:
        generate(n_steps=n_steps, output_path=f"sample_images/steps_{n_steps}.png")