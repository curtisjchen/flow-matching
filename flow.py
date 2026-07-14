import torch
import torch.nn.functional as F

def flow_matching_loss(model, x_1):
    x_0 = torch.randn_like(x_1)
    t = torch.rand(x_1.shape[0], device=x_1.device)
    t_reshaped = t.reshape(-1, 1, 1, 1)
    x_t =  (1-t_reshaped) * x_0 + t_reshaped * x_1
    v_star = x_1 - x_0 # this is the target
    v_theta = model(x_t, t, t)# this is our model 
    return F.mse_loss(v_theta, v_star)
