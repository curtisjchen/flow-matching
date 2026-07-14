import torch
import torch.nn.functional as F

def mean_flow_loss(model, x_1, p_rt=0.5):
    device = x_1.device
    b = x_1.shape[0]
 
    x_0 = torch.randn_like(x_1)
    t = torch.rand(b, device=device)
 
    # r = t with prob p_rt, else r ~ U(0, t)
    r = torch.where(torch.rand(b, device=device) < p_rt, t, torch.rand(b, device=device) * t)
 
    r_ = r.reshape(-1, 1, 1, 1)
    z_r = (1 - r_) * x_0 + r_ * x_1
    v = x_1 - x_0
 
    def f(z, r_arg, t_arg):
        return model(z, r_arg, t_arg)
 
    # TODO: compute u, du_dr via torch.func.jvp(f, primals, tangents)
    primals  = (z_r, r, t)
    tangents = (v, torch.ones_like(r), torch.zeros_like(t))
    u, du_dr = torch.func.jvp(f, primals, tangents)

 
    t_minus_r = (t - r).reshape(-1, 1, 1, 1)
    u_target = v + t_minus_r * du_dr
 
    return F.mse_loss(u, u_target.detach())

if __name__ == "__main__":
    from models.unet import UNet

    model = UNet()
    x_1 = torch.randn(4, 1, 28, 28)  # fake batch of "images"

    loss = mean_flow_loss(model, x_1)
    print(loss)