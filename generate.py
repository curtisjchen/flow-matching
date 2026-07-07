import torch, torchvision
from models.unet import UNet
from solver import euler_solve
import os
import yaml

def generate(config_path="configs/unet_mnist_large.yaml", n_steps=150, checkpoint_path="checkpoints/unet_mnist_large_epoch_20.pt", samples=16, output_path="sample_images/"):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    model = UNet(time_in=config["model"]["time_in"],
                time_out=config["model"]["time_out"],
                down_in_1=config["model"]["down_in_1"],
                down_in_2=config["model"]["down_in_2"],
                down_out_1=config["model"]["down_out_1"],
                down_out_2=config["model"]["down_out_2"],
                prefinal=config["model"]["prefinal"])
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
    for n_steps in [5, 25, 150]:
        generate(n_steps=n_steps, output_path=f"sample_images/steps_{n_steps}.png")