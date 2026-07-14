import torch
def euler_solve(model, N, shape):
    device = next(model.parameters()).device
    with torch.inference_mode():
        timesteps = torch.linspace(0, 1, N, device=device)
        noise_sample = torch.randn(shape, device=device)
        dt = 1 / (N - 1) if N > 1 else 1.0
        for t in timesteps:
            t_batch = torch.full((shape[0],), t.item(), device=device)
            dv = model(noise_sample, t_batch, t_batch)
            noise_sample += dv * dt
        return noise_sample

def one_step_sample(model, shape, device=None):
    device = device or next(model.parameters()).device
    model.eval()
    with torch.inference_mode():
        x_0 = torch.randn(shape, device=device)
        r = torch.zeros(shape[0], device=device)
        t = torch.ones(shape[0], device=device)
        u = model(x_0, r, t)
        x_1 = x_0 + u
    return x_1