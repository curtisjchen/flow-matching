import torch
import torch.nn.functional as F
from torch.nn.attention import sdpa_kernel, SDPBackend

def improved_velocity_from_mean_velocity(
    mean_velocity: torch.Tensor,
    d_mean_velocity_dr: torch.Tensor,
    interval: torch.Tensor,
) -> torch.Tensor:
    """Convert a mean velocity to an instantaneous velocity.

    This project parameterizes paths from noise at time 0 to data at time 1,
    so the iMF identity is v(z_r, r) = u(z_r, r, t) - (t-r) D_r u.
    """
    return mean_velocity - interval * d_mean_velocity_dr


def sample_ordered_times(
    batch_size,
    device,
    time_sampler="uniform",
    logit_normal_mean=-0.4,
    logit_normal_std=1.0,
):
    """Sample a pair of scalar times satisfying r <= t for every example."""
    if time_sampler == "uniform":
        first = torch.rand(batch_size, device=device)
        second = torch.rand(batch_size, device=device)
    elif time_sampler == "logit_normal":
        first = torch.sigmoid(
            torch.randn(batch_size, device=device) * logit_normal_std
            + logit_normal_mean
        )
        second = torch.sigmoid(
            torch.randn(batch_size, device=device) * logit_normal_std
            + logit_normal_mean
        )
    else:
        raise ValueError(
            f"Unsupported time_sampler={time_sampler!r}; use 'uniform' or 'logit_normal'."
        )
    return torch.minimum(first, second), torch.maximum(first, second)


def _prepare_labels(labels, model, p_uncond, device):
    raw_model = getattr(model, "module", model)
    raw_model = getattr(raw_model, "_orig_mod", raw_model)
    null_class_idx = raw_model.null_class_idx
    drop_mask = torch.rand(labels.shape[0], device=device) < p_uncond
    train_labels = torch.where(drop_mask, null_class_idx, labels)
    pure_null_labels = torch.full_like(labels, null_class_idx)
    return train_labels, pure_null_labels


def _sample_w(labels, w_min, w_max, device):
    w = w_min + torch.rand(labels.shape[0], device=device) * (w_max - w_min)
    return w, w.view(-1, 1, 1, 1)


def _make_cfg_fn(model, train_labels, pure_null_labels, w, w_reshaped):
    """Create f(z, r, t), the classifier-free-guided mean-velocity model."""
    raw_model = getattr(model, "module", model)
    raw_model = getattr(raw_model, "_orig_mod", raw_model)
    def f(z_in, r_in, t_in):
        z_double = torch.cat([z_in, z_in], dim=0)
        r_double = torch.cat([r_in, r_in], dim=0)
        t_double = torch.cat([t_in, t_in], dim=0)
        labels_double = torch.cat([train_labels, pure_null_labels], dim=0)
        w_double = torch.cat([w, w], dim=0)

        output_double = raw_model(
            z_double, r_double, t_double, w=w_double, class_labels=labels_double
        )
        output_cond, output_uncond = output_double.chunk(2, dim=0)
        return output_uncond + w_reshaped * (output_cond - output_uncond)

    return f

def _make_standard_fn(model, train_labels, w):
    """Create f(z, r, t) using a standard, single-batch forward pass for fast training."""
    raw_model = getattr(model, "module", model)
    raw_model = getattr(raw_model, "_orig_mod", raw_model)
    
    def f(z_in, r_in, t_in):
        # Single forward pass at batch size N (No double batching!)
        return raw_model(z_in, r_in, t_in, w=w, class_labels=train_labels)

    return f

def _adaptive_velocity_loss(prediction, target, power, eps):
    """Paper-style per-example adaptive L2 loss plus an interpretable raw MSE."""
    squared_error = (prediction - target.detach()).square().flatten(1).sum(dim=1)
    adaptive_weight = (squared_error.detach() + eps).pow(power)
    loss = (squared_error / adaptive_weight).mean()
    raw_mse = (squared_error / target[0].numel()).mean().detach()
    return loss, raw_mse


def mean_flow_loss(
    model,
    x_1,
    labels,
    p_rt=0.5,
    p_uncond=0.1,
    w_min=1.0,
    w_max=5.0,
    adaptive_loss_power=1.0,
    adaptive_loss_eps=1e-2,
    time_sampler="uniform",
    logit_normal_mean=-0.4,
    logit_normal_std=1.0,
    clamp_d_dr=None,
    cfg_aware_loss=True
):

    device = x_1.device
    batch_size = x_1.shape[0]
    x_0 = torch.randn_like(x_1)

    r, t = sample_ordered_times(
        batch_size, device, time_sampler, logit_normal_mean, logit_normal_std
    )
    r = torch.where(torch.rand(batch_size, device=device) < p_rt, t, r)

    r_image = r.view(-1, 1, 1, 1)
    z_r = (1 - r_image) * x_0 + r_image * x_1
    velocity = x_1 - x_0

    train_labels, pure_null_labels = _prepare_labels(labels, model, p_uncond, device)
    w, w_reshaped = _sample_w(labels, w_min, w_max, device)

    if cfg_aware_loss:
        f = _make_cfg_fn(model, train_labels, pure_null_labels, w, w_reshaped)
    else:
        f = _make_standard_fn(model, train_labels, w)

    with sdpa_kernel(SDPBackend.MATH):
        mean_velocity, d_mean_velocity_dr = torch.func.jvp(
            f,
            (z_r, r, t),
            (velocity, torch.ones_like(r), torch.zeros_like(t)),
        )

    diag_max_d_dr = d_mean_velocity_dr.detach().abs().max()

    if clamp_d_dr is not None:
        d_mean_velocity_dr = torch.clamp(d_mean_velocity_dr, min=-clamp_d_dr, max=clamp_d_dr)

    interval = (t - r).view(-1, 1, 1, 1)
    target_mean_velocity = velocity + interval * d_mean_velocity_dr
    loss, raw_mse = _adaptive_velocity_loss(
        mean_velocity, target_mean_velocity, adaptive_loss_power, adaptive_loss_eps
    )

    return loss, raw_mse, {"max_abs_d_dr": diag_max_d_dr}


def improved_mean_flow_loss(
    model,
    x_1,
    labels,
    p_rt=0.5,
    p_uncond=0.1,
    w_min=1.0,
    w_max=5.0,
    adaptive_loss_power=1.0,
    adaptive_loss_eps=1e-2,
    time_sampler="uniform",
    logit_normal_mean=-0.4,
    logit_normal_std=1.0,
    clamp_d_dr=None,
    cfg_aware_loss=True
):
    device = x_1.device
    batch_size = x_1.shape[0]
    x_0 = torch.randn_like(x_1)

    r, t = sample_ordered_times(
        batch_size, device, time_sampler, logit_normal_mean, logit_normal_std
    )
    r = torch.where(torch.rand(batch_size, device=device) < p_rt, t, r)

    r_image = r.view(-1, 1, 1, 1)
    z_r = (1 - r_image) * x_0 + r_image * x_1
    target_v = x_1 - x_0  # ground truth; stays the regression target

    train_labels, pure_null_labels = _prepare_labels(labels, model, p_uncond, device)
    w, w_reshaped = _sample_w(labels, w_min, w_max, device)
    if cfg_aware_loss:
        f = _make_cfg_fn(model, train_labels, pure_null_labels, w, w_reshaped)
    else:
        f = _make_standard_fn(model, train_labels, w)

    boundary_velocity = f(z_r, r, r).detach()

    with sdpa_kernel(SDPBackend.MATH):
        mean_velocity, d_mean_velocity_dr = torch.func.jvp(
            f,
            (z_r, r, t),
            (boundary_velocity, torch.ones_like(r), torch.zeros_like(t)),
        )

    diag_max_d_dr = d_mean_velocity_dr.detach().abs().max()

    if clamp_d_dr is not None:
        d_mean_velocity_dr = torch.clamp(d_mean_velocity_dr, min=-clamp_d_dr, max=clamp_d_dr)

    # Paper: "the stop-gradient is part of the prediction function V_theta,
    # not the regression target" -- same detach placement as mean_flow_loss.
    interval = (t - r).view(-1, 1, 1, 1)
    v_pred = improved_velocity_from_mean_velocity(
        mean_velocity, d_mean_velocity_dr.detach(), interval
    )

    loss, raw_mse = _adaptive_velocity_loss(
        v_pred, target_v, adaptive_loss_power, adaptive_loss_eps
    )

    return loss, raw_mse, {"max_abs_d_dr": diag_max_d_dr}