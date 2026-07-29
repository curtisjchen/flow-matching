import torch
import torch.nn.functional as F

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
        # Stack everything to size (2B, ...)
        z_double = torch.cat([z_in, z_in], dim=0)
        r_double = torch.cat([r_in, r_in], dim=0)
        t_double = torch.cat([t_in, t_in], dim=0)
        labels_double = torch.cat([train_labels, pure_null_labels], dim=0)
        
        # NOTE: If your UNet's forward pass now requires 'w' because 
        # of the WEmbedding, stack it and pass it too!
        w_double = torch.cat([w, w], dim=0)

        # Single optimized pass
        v_double = model(z_double, r_double, t_double, labels_double, w_double)
        
        # Split and apply the CFG formula
        v_cond, v_uncond = v_double.chunk(2, dim=0)
        return v_uncond + w_reshaped * (v_cond - v_uncond)

    primals  = (z_r, r, t)
    tangents = (v, torch.ones_like(r), torch.zeros_like(t))
    u, du_dr = torch.func.jvp(f, primals, tangents)

    # 1. Clamp the derivative to prevent exploding targets
    du_dr = torch.clamp(du_dr, min=-20.0, max=20.0)

    t_minus_r = (t - r).reshape(-1, 1, 1, 1)
    u_target = v + t_minus_r * du_dr

    return F.smooth_l1_loss(u, u_target.detach())

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

    # 2. Setup W
    u = torch.rand(labels.shape[0], device=device)
    w = w_min + u * (w_max - w_min)
    w_reshaped = w.view(-1, 1, 1, 1)

    # 3. The Custom JVP Wrapper
    def f(z_in, r_in, t_in):
        # Stack everything to size (2B, ...)
        z_double = torch.cat([z_in, z_in], dim=0)
        r_double = torch.cat([r_in, r_in], dim=0)
        t_double = torch.cat([t_in, t_in], dim=0)
        labels_double = torch.cat([train_labels, pure_null_labels], dim=0)
        
        # NOTE: If your UNet's forward pass now requires 'w' because 
        # of the WEmbedding, stack it and pass it too!
        w_double = torch.cat([w, w], dim=0)

        # Single optimized pass
        v_double = model(z_double, r_double, t_double, labels_double, w_double)
        
        # Split and apply the CFG formula
        v_cond, v_uncond = v_double.chunk(2, dim=0)
        return v_uncond + w_reshaped * (v_cond - v_uncond)
 
    v_theta = f(z_r, r, r)
 
    primals  = (z_r, r, t)
    tangents = (v_theta, torch.ones_like(r), torch.zeros_like(t))
    model_output, du_dr = torch.func.jvp(f, primals, tangents)
 
    du_dr = torch.clamp(du_dr, min=-20.0, max=20.0)
    t_minus_r = (t - r).reshape(-1, 1, 1, 1)
    V_theta = model_output + t_minus_r * du_dr.detach()
 
    return F.smooth_l1_loss(V_theta, v_star)

if __name__ == "__main__":
    from models.unet import UNet
    import torch
    
    # 1. Initialize your model
    model = UNet(
        down_in_1=1, down_in_2=64, down_out_1=64, down_out_2=256,
        prefinal=32, time_in=128, time_out=256
    )
    
    if not hasattr(model, 'null_class_idx'):
        model.null_class_idx = 10 
        
    batch_size = 4
    
    # 2. Create dummy image data (like a batch of 4 MNIST images)
    x_1 = torch.randn(batch_size, 1, 28, 28)
    
    # 3. Create dummy labels (random integers between 0 and 9)
    labels = torch.randint(low=0, high=10, size=(batch_size,))
    
    print(f"Testing with batch size {batch_size}...")
    print(f"Original labels: {labels}")
    
    # 4. Test Mean Flow Loss
    try:
        loss1 = mean_flow_loss(model, x_1, labels)
        print(f"Mean Flow Loss: {loss1.item():.4f} - SUCCESS")
    except Exception as e:
        print(f"Mean Flow Loss FAILED: {e}")
        
    # 5. Test Improved Mean Flow Loss
    try:
        loss2 = imf_loss(model, x_1, labels)
        print(f"Improved Mean Flow Loss: {loss2.item():.4f} - SUCCESS")
    except Exception as e:
        print(f"Improved Mean Flow Loss FAILED: {e}")