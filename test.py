import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from utils import build_model
from mean_flow import mean_flow_loss
import os
import yaml

with open("configs/dit_mnist_imf_xl_compile.yaml", "r") as f:
    config = yaml.safe_load(f)

dist.init_process_group(backend="nccl")
local_rank = int(os.environ["LOCAL_RANK"])
torch.cuda.set_device(local_rank)
device = torch.device(f"cuda:{local_rank}")

model = build_model(config).to(device)
model = DDP(model, device_ids=[local_rank], static_graph=True)

torch.cuda.reset_peak_memory_stats(device)

for step in range(10):
    x = torch.randn(512, 1, 28, 28, device=device)
    labels = torch.randint(0, 10, (64,), device=device)

    loss, raw_mse, diag = mean_flow_loss(model=model, x_1=x, labels=labels)  # note: model, the DDP-wrapped object
    loss.backward()

    peak = torch.cuda.max_memory_allocated(device) / (1024**3)
    print(f"[rank {local_rank}] step {step} | loss {loss.item():.4f} | peak mem {peak:.2f} GB")