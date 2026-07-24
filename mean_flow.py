import torch
import torch.nn.functional as F

def mean_flow_loss(model, x_1, p_rt=0.75):
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

    def f(z, r_arg, t_arg):
        return model(z, r_arg, t_arg)

    primals  = (z_r, r, t)
    tangents = (v, torch.ones_like(r), torch.zeros_like(t))
    u, du_dr = torch.func.jvp(f, primals, tangents)

    t_minus_r = (t - r).reshape(-1, 1, 1, 1)
    u_target = v + t_minus_r * du_dr

    return F.mse_loss(u, u_target.detach())

if __name__ == "__main__":
    from models.unet import UNet

    model = UNet(down_in_1=1, down_in_2=64, down_out_1=64, down_out_2=256,
                 prefinal=32, time_in=128, time_out=256)
    x_1 = torch.randn(4, 1, 28, 28)

    loss = mean_flow_loss(model, x_1)
    print(loss)