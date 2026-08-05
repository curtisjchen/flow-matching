import torch
from utils import build_model
from mean_flow import mean_flow_loss
import yaml

with open("configs/dit_mnist_mf_large_gpu.yaml", "r") as f:
    config = yaml.safe_load(f)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
print(f"GPU: {torch.cuda.get_device_name(0)}")

model = build_model(config).to(device)
x = torch.randn(4, 1, 28, 28, device=device)
labels = torch.randint(0, config["model"]["num_classes"], (4,), device=device)

with torch.autocast(device_type='cuda', dtype=torch.float16):
    loss, raw_mse, diag = mean_flow_loss(model=model, x_1=x, labels=labels)
    print("loss dtype inside autocast:", loss.dtype)

loss.backward()
print("OK (fp16 autocast):", loss.item())