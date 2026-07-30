import torch
import torch.nn.functional as F
from torch.nn.attention import sdpa_kernel, SDPBackend

def mean_flow_loss(model, x_1, labels, p_rt=0.5, p_uncond=0.1, w_min=1.0, w_max=5.0):
    device = x_1.device
    b = x_1.shape[0]

    x_0 = torch.randn_like(x_1)

    # sample r, t independently, then swap to enforce r <= t
    a = torch.rand(b, device=device)
    c = torch.rand(b, device=device)
    t = torch.maximum(a, c)
    r = torch.minimum(a, c)

    # cap the proportion of r != t: force r = t for p_rt fraction of the batch
    force_eq = torch.rand(b, device=device) < p_rt
    r = torch.where(force_eq, t, r)

    r_ = r.reshape(-1, 1, 1, 1)
    z_r = (1 - r_) * x_0 + r_ * x_1
    v = x_1 - x_0

    #drop labels
    drop_mask = torch.rand(labels.shape[0], device=device) < p_uncond
    train_labels = torch.where(drop_mask, model.null_class_idx, labels)
    pure_null_labels = torch.full_like(labels, model.null_class_idx)

    #sample w
    u = torch.rand(labels.shape[0], device=labels.device)
    w = w_min + u * (w_max - w_min)
    w_reshaped = w.view(-1, 1, 1, 1)

    def f(z_in, r_in, t_in):
        z_double = torch.cat([z_in, z_in], dim=0)
        r_double = torch.cat([r_in, r_in], dim=0)
        t_double = torch.cat([t_in, t_in], dim=0)
        labels_double = torch.cat([train_labels, pure_null_labels], dim=0)

        w_double = torch.cat([w, w], dim=0)

        v_double = model(
            z_double, r_double, t_double, w=w_double, class_labels=labels_double
        )
        
        v_cond, v_uncond = v_double.chunk(2, dim=0)
        return v_uncond + w_reshaped * (v_cond - v_uncond)

    primals  = (z_r, r, t)
    tangents = (v, torch.ones_like(r), torch.zeros_like(t))
    with torch.autocast(device_type="cuda", enabled=False):
        primals_fp32 = tuple(p.float() for p in primals)
        tangents_fp32 = tuple(t_.float() for t_ in tangents)
        u, du_dr = torch.func.jvp(f, primals_fp32, tangents_fp32)

    u = u.float()
    du_dr = torch.nan_to_num(du_dr, nan=0.0, posinf=100.0, neginf=-100.0)
    du_dr = du_dr.detach()
    du_dr_max = du_dr.abs().max().item()
    du_dr = torch.clamp(du_dr, min=-100.0, max=100.0)

    t_minus_r = (t - r).reshape(-1, 1, 1, 1)
    u_target = v + t_minus_r * du_dr

    loss = F.smooth_l1_loss(u, u_target.detach())
    return loss, du_dr_max

def imf_loss(model, x_1, labels, p_rt=0.5, p_uncond=0.1, w_min=1.0, w_max=5.0):    
    device = x_1.device
    b = x_1.shape[0]
 
    x_0 = torch.randn_like(x_1)
 
    a = torch.rand(b, device=device)
    c = torch.rand(b, device=device)
    t = torch.maximum(a, c)
    r = torch.minimum(a, c)
 
    force_eq = torch.rand(b, device=device) < p_rt
    r = torch.where(force_eq, t, r)
 
    r_ = r.reshape(-1, 1, 1, 1)
    z_r = (1 - r_) * x_0 + r_ * x_1
    v_star = x_1 - x_0

    drop_mask = torch.rand(labels.shape[0], device=device) < p_uncond
    train_labels = torch.where(drop_mask, model.null_class_idx, labels)
    pure_null_labels = torch.full_like(labels, model.null_class_idx)

    u = torch.rand(labels.shape[0], device=device)
    w = w_min + u * (w_max - w_min)
    w_reshaped = w.view(-1, 1, 1, 1)

    def f(z_in, r_in, t_in):
        z_double = torch.cat([z_in, z_in], dim=0)
        r_double = torch.cat([r_in, r_in], dim=0)
        t_double = torch.cat([t_in, t_in], dim=0)
        labels_double = torch.cat([train_labels, pure_null_labels], dim=0)
        
        w_double = torch.cat([w, w], dim=0)

        v_double = model(
            z_double, r_double, t_double, w=w_double, class_labels=labels_double
        )
        
        v_cond, v_uncond = v_double.chunk(2, dim=0)
        return v_uncond + w_reshaped * (v_cond - v_uncond)
 
    v_theta = f(z_r, r, r)
 
    primals  = (z_r, r, t)
    tangents = (v_theta, torch.ones_like(r), torch.zeros_like(t))
    with torch.autocast(device_type="cuda", enabled=False):
        primals_fp32 = tuple(p.float() for p in primals)
        tangents_fp32 = tuple(t_.float() for t_ in tangents)
        with sdpa_kernel(SDPBackend.MATH):
            model_output, du_dr = torch.func.jvp(f, primals_fp32, tangents_fp32)

    model_output = model_output.float()

    du_dr = torch.nan_to_num(du_dr, nan=0.0, posinf=100.0, neginf=-100.0)
    du_dr = du_dr.detach()
    du_dr_max = du_dr.abs().max().item()
    du_dr = torch.clamp(du_dr, min=-100.0, max=100.0)

    t_minus_r = (t - r).reshape(-1, 1, 1, 1)
    V_theta = model_output + t_minus_r * du_dr

    loss = F.smooth_l1_loss(V_theta, v_star, beta = 2.0)
    return loss, du_dr_max

if __name__ == "__main__":
    from models.unet import UNet
    import torch
    
    model = UNet(
        down_in_1=1, down_in_2=64, down_out_1=64, down_out_2=256,
        prefinal=32, time_in=128, time_out=256
    )
    
    if not hasattr(model, 'null_class_idx'):
        model.null_class_idx = 10 
        
    batch_size = 4
    
    x_1 = torch.randn(batch_size, 1, 28, 28)
    
    labels = torch.randint(low=0, high=10, size=(batch_size,))
    
    print(f"Testing with batch size {batch_size}...")
    print(f"Original labels: {labels}")
    
    try:
        loss1 = mean_flow_loss(model, x_1, labels)
        print(f"Mean Flow Loss: {loss1.item():.4f} - SUCCESS")
    except Exception as e:
        print(f"Mean Flow Loss FAILED: {e}")
        
    try:
        loss2 = imf_loss(model, x_1, labels)
        print(f"Improved Mean Flow Loss: {loss2.item():.4f} - SUCCESS")
    except Exception as e:
        print(f"Improved Mean Flow Loss FAILED: {e}")
