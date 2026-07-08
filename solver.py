import torch
def euler_solve(model, N, shape):
    device = next(model.parameters()).device
    with torch.inference_mode():
        timesteps = torch.linspace(0, 1, N, device=device)
        noise_sample = torch.randn(shape, device=device)
        dt = 1 / (N - 1) if N > 1 else 1.0
        for t in timesteps:
            t_batch = torch.full((shape[0],), t.item(), device=device)
            dv = model(noise_sample, t_batch)
            noise_sample += dv * dt
        return noise_sample