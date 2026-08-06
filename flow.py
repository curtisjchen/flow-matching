import torch
import torch.nn.functional as F

def flow_matching_loss(model, x_1, labels, p_uncond=0.1, w_min=1.0, w_max=5.0):
    device = x_1.device
    b = x_1.shape[0]

    x_0 = torch.randn_like(x_1)
    t = torch.rand(b, device=device)
    
    t_reshaped = t.reshape(-1, 1, 1, 1)
    x_t = (1 - t_reshaped) * x_0 + t_reshaped * x_1
    v_target = x_1 - x_0

    drop_mask = torch.rand(b, device=device) < p_uncond
    null_idx = model.module.null_class_idx if hasattr(model, "module") else model.null_class_idx
    train_labels = torch.where(drop_mask, null_idx, labels)

    u = torch.rand(b, device=device)
    w = w_min + u * (w_max - w_min)

    v_pred = model(x_t, t, t, w=w, class_labels=train_labels)

    return F.mse_loss(v_pred, v_target)
