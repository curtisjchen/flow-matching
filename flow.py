import torch
import torch.nn.functional as F
from mean_flow import _prepare_labels, _sample_w, _make_cfg_fn, _make_standard_fn  # adjust import path

def flow_matching_loss(model, x_1, labels, p_uncond=1.0, w_min=1.0, w_max=5.0, cfg_aware_loss=True):
    device = x_1.device
    b = x_1.shape[0]

    x_0 = torch.randn_like(x_1)
    t = torch.rand(b, device=device)

    t_reshaped = t.reshape(-1, 1, 1, 1)
    x_t = (1 - t_reshaped) * x_0 + t_reshaped * x_1
    v_target = x_1 - x_0

    train_labels, pure_null_labels = _prepare_labels(labels, model, p_uncond, device)

    if cfg_aware_loss:
        w, w_reshaped = _sample_w(labels, w_min, w_max, device)
        f = _make_cfg_fn(model, train_labels, pure_null_labels, w, w_reshaped)
    else:
        # Pure unconditional: w is architecturally required but carries no
        # signal, so fix it to a constant instead of sampling a range the
        # model has no way to learn to use.
        w = torch.ones(b, device=device)
        f = _make_standard_fn(model, train_labels, w)

    v_pred = f(x_t, t, t)
    return F.mse_loss(v_pred, v_target)
