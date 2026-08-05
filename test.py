import torch
from utils import build_model
from mean_flow import mean_flow_loss
import yaml

with open("configs/dit_mnist_mf_large_gpu.yaml", "r") as f:
    config = yaml.safe_load(f)

device_type = 'cuda' if torch.cuda.is_available() else 'cpu'
device = torch.device(device_type)
print(f"Using device: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")

model = build_model(config).to(device)
x = torch.randn(4, 1, 28, 28, device=device)
labels = torch.randint(0, config["model"]["num_classes"], (4,), device=device)

with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
    loss, raw_mse, diag = mean_flow_loss(model=model, x_1=x, labels=labels)
    print("loss dtype inside autocast:", loss.dtype)  # bf16 autocast keeps reductions in fp32 by default, so this being float32 is expected and fine

loss.backward()
print("OK:", loss.item())

# --- Diagnostics: confirm bf16 casting actually reached model internals ---
if torch.cuda.is_available():
    with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
        with torch.no_grad():
            probe_out = model.patch_embed(x)
            print("patch_embed output dtype:", probe_out.dtype)  # expect torch.bfloat16 if casting is active

    # --- Diagnostics: which SDPA backend does this model's attention actually hit? ---
    from torch.profiler import profile, ProfilerActivity

    h = torch.randn(4, model.patch_embed.num_patches, config["model"]["hidden_dim"], device=device, dtype=torch.bfloat16)
    block = model.ditblocks[0]

    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        with torch.no_grad():
            _ = block.attention(h, h, h, need_weights=False)[0]
    print("\n--- Attention backend trace (outside JVP) ---")
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=5))

    # --- Confirm MATH backend is actually forced inside the JVP-traced region ---
    from torch.nn.attention import sdpa_kernel, SDPBackend
    import torch.nn.functional as F

    q = k = v = torch.randn(2, config["model"]["num_heads"], 8,
                             config["model"]["hidden_dim"] // config["model"]["num_heads"],
                             device=device, dtype=torch.bfloat16)
    for backend in [SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH]:
        try:
            with sdpa_kernel(backend):
                _ = F.scaled_dot_product_attention(q, k, v)
            print(f"{backend}: supported on this GPU/shape")
        except Exception as e:
            print(f"{backend}: NOT supported — {e}")
else:
    print("\n(Skipping GPU-specific dtype/backend diagnostics — no CUDA device available)")