import torch
def euler_solve(model, N, shape):
    if N < 1:
        raise ValueError("N must be at least 1.")

    device = next(model.parameters()).device
    with torch.inference_mode():
        sample = torch.randn(shape, device=device)
        dt = 1.0 / N

        for i in range(N):
            t = i * dt
            t_batch = torch.full((shape[0],), t, device=device)
            sample = sample + dt * model(sample, t_batch, t_batch)

    return sample

def one_step_sample(model, shape, device=None):
    device = device or next(model.parameters()).device
    with torch.inference_mode():
        x_0 = torch.randn(shape, device=device)
        r = torch.zeros(shape[0], device=device)
        t = torch.ones(shape[0], device=device)
        u = model(x_0, r, t)
        x_1 = x_0 + u
    return x_1

def mean_flow_multistep_sample(model, N, shape, device=None):
    """Multi-step MeanFlow sampling: splits [0,1] into N equal intervals and
    jumps across each one using the *average* velocity for that interval
    (model queried with r < t), rather than the instantaneous velocity
    euler_solve uses (model queried with r == t)."""
    if N < 1:
        raise ValueError("N must be at least 1.")
    device = device or next(model.parameters()).device
 
    with torch.inference_mode():
        sample = torch.randn(shape, device=device)
        boundaries = torch.linspace(0.0, 1.0, N + 1, device=device)
 
        for i in range(N):
            r_val, t_val = boundaries[i].item(), boundaries[i + 1].item()
            r = torch.full((shape[0],), r_val, device=device)
            t = torch.full((shape[0],), t_val, device=device)
            u = model(sample, r, t)
            sample = sample + (t_val - r_val) * u
 
    return sample
