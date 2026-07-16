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
    model.eval()
    with torch.inference_mode():
        x_0 = torch.randn(shape, device=device)
        r = torch.zeros(shape[0], device=device)
        t = torch.ones(shape[0], device=device)
        u = model(x_0, r, t)
        x_1 = x_0 + u
    return x_1