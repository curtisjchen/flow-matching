import torch
def euler_solve(model, N, shape):
    with torch.inference_mode():
        timesteps = torch.linspace(0, 1, N)[:-1]
        noise_sample = torch.randn(shape)
        dt = 1 / N
        for t in timesteps:
            t_batch = torch.full((shape[0],), t.item())
            dv = model(noise_sample, t_batch)
            noise_sample += dv * dt
        
        return noise_sample