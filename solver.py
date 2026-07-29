import torch
import torch

def euler_solve(model, N, shape, labels, w_val, null_class_idx):
    if N < 1:
        raise ValueError("N must be at least 1.")

    device = next(model.parameters()).device
    b = shape[0]

    null_labels = torch.full((b,), null_class_idx, dtype=torch.long, device=device)
    labels_double = torch.cat([labels, null_labels], dim=0)
    
    w_tensor = torch.full((b,), w_val, device=device)
    w_double = torch.cat([w_tensor, w_tensor], dim=0)
    w_reshaped = w_tensor.view(-1, 1, 1, 1)

    with torch.inference_mode():
        sample = torch.randn(shape, device=device)
        dt = 1.0 / N

        for i in range(N):
            t = i * dt
            t_batch = torch.full((b,), t, device=device)
            
            sample_double = torch.cat([sample, sample], dim=0)
            t_double = torch.cat([t_batch, t_batch], dim=0)
            
            v_double = model(sample_double, t_double, t_double, labels_double, w_double)
            
            v_cond, v_uncond = v_double.chunk(2, dim=0)
            
            v_cfg = v_uncond + w_reshaped * (v_cond - v_uncond)

            sample = sample + dt * v_cfg

    return sample

def mean_flow_multistep_sample(model, N, shape, labels, w_val, null_class_idx):
    if N < 1:
        raise ValueError("N must be at least 1.")

    device = next(model.parameters()).device
    b = shape[0]

    null_labels = torch.full((b,), null_class_idx, dtype=torch.long, device=device)
    labels_double = torch.cat([labels, null_labels], dim=0)
    
    w_tensor = torch.full((b,), w_val, device=device)
    w_double = torch.cat([w_tensor, w_tensor], dim=0)
    w_reshaped = w_tensor.view(-1, 1, 1, 1)
 
    with torch.inference_mode():
        sample = torch.randn(shape, device=device)
        boundaries = torch.linspace(0.0, 1.0, N + 1, device=device)
 
        for i in range(N):
            r_val, t_val = boundaries[i].item(), boundaries[i + 1].item()
            r = torch.full((b,), r_val, device=device)
            t = torch.full((b,), t_val, device=device)
            
            sample_double = torch.cat([sample, sample], dim=0)
            r_double = torch.cat([r, r], dim=0)
            t_double = torch.cat([t, t], dim=0)
            
            v_double = model(sample_double, r_double, t_double, labels_double, w_double)
            
            v_cond, v_uncond = v_double.chunk(2, dim=0)
            
            v_cfg = v_uncond + w_reshaped * (v_cond - v_uncond)
            
            sample = sample + (t_val - r_val) * v_cfg

    return sample
