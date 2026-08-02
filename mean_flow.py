import torch
import torch.nn.functional as F


def improved_velocity_from_mean_velocity(
    mean_velocity: torch.Tensor,
    d_mean_velocity_dr: torch.Tensor,
    interval: torch.Tensor,
) -> torch.Tensor:
    """Convert a mean-velocity prediction to an instantaneous-velocity prediction.

    This project parameterizes the path from noise at time 0 to data at time 1.
    For that convention, the MeanFlow identity is

        v(z_r, r) = u(z_r, r, t) - (t - r) D_r u(z_r, r, t).

    (The sign is reversed in implementations that parameterize the path from
    data to noise.)  Keeping this conversion separate makes the time
    convention explicit and prevents a very easy-to-miss sign error.
    """
    return mean_velocity - interval * d_mean_velocity_dr


def sample_ordered_times(
    batch_size,
    device,
    time_sampler="uniform",
    logit_normal_mean=-0.4,
    logit_normal_std=1.0,
):
    """Sample r <= t using the paper's logit-normal or a uniform schedule."""
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
    return_raw_mse=False,
):
    device = x_1.device
    b = x_1.shape[0]

    x_0 = torch.randn_like(x_1)

    r, t = sample_ordered_times(
        b, device, time_sampler, logit_normal_mean, logit_normal_std
    )

    # cap the proportion of r != t: force r = t for p_rt fraction of the batch
    force_eq = torch.rand(b, device=device) < p_rt
    r = torch.where(force_eq, t, r)
    return r, t


def _prepare_labels(labels, model, p_uncond, device):
    drop_mask = torch.rand(labels.shape[0], device=device) < p_uncond
    train_labels = torch.where(drop_mask, model.null_class_idx, labels)
    pure_null_labels = torch.full_like(labels, model.null_class_idx)
    return train_labels, pure_null_labels


def _sample_w(labels, w_min, w_max, device):
    u = torch.rand(labels.shape[0], device=device)
    w = w_min + u * (w_max - w_min)
    return w, w.view(-1, 1, 1, 1)


def _make_cfg_fn(model, train_labels, pure_null_labels, w, w_reshaped):
    """Builds a CFG-wrapped forward closure: f(z, r, t) -> guided velocity."""
    def f(z_in, r_in, t_in):
        z_double = torch.cat([z_in, z_in], dim=0)
        r_double = torch.cat([r_in, r_in], dim=0)
        t_double = torch.cat([t_in, t_in], dim=0)
        labels_double = torch.cat([train_labels, pure_null_labels], dim=0)
        w_double = torch.cat([w, w], dim=0)

        v_double = model(z_double, r_double, t_double, w=w_double, class_labels=labels_double)
        v_cond, v_uncond = v_double.chunk(2, dim=0)
        return v_uncond + w_reshaped * (v_cond - v_uncond)
    return f


def _clean_du_dr(du_dr, clamp_val=20.0):
    du_dr = torch.nan_to_num(du_dr, nan=0.0, posinf=clamp_val, neginf=-clamp_val)
    du_dr = du_dr.detach()
    du_dr_max = du_dr.abs().max().item()
    du_dr = torch.clamp(du_dr, min=-clamp_val, max=clamp_val)
    return du_dr, du_dr_max

def mean_flow_loss(model, x_1, labels, p_rt=0.5, p_uncond=0.1, w_min=1.0, w_max=5.0, clamp_val=100.0):
    device = x_1.device
    b = x_1.shape[0]

    x_0 = torch.randn_like(x_1)
    r, t = _sample_r_t(b, device, p_rt)

    r_ = r.reshape(-1, 1, 1, 1)
    z_r = (1 - r_) * x_0 + r_ * x_1
    v = x_1 - x_0

    train_labels, pure_null_labels = _prepare_labels(labels, model, p_uncond, device)
    w, w_reshaped = _sample_w(labels, w_min, w_max, device)
    f = _make_cfg_fn(model, train_labels, pure_null_labels, w, w_reshaped)

    primals = (z_r, r, t)
    tangents = (v, torch.ones_like(r), torch.zeros_like(t))
    u, du_dr = torch.func.jvp(f, primals, tangents)

    # Paper replication: do not clip the JVP.
    # du_dr = torch.clamp(du_dr, min=-20.0, max=20.0)

    t_minus_r = (t - r).reshape(-1, 1, 1, 1)
    u_target = v + t_minus_r * du_dr

    residual = u - u_target.detach()
    squared_error = residual.square().flatten(1).sum(dim=1)
    adaptive_weight = (squared_error.detach() + adaptive_loss_eps).pow(
        adaptive_loss_power
    )
    loss = (squared_error / adaptive_weight).mean()

    # Previous experiment, retained for quick A/B comparison:
    # return F.smooth_l1_loss(u, u_target.detach())
    if return_raw_mse:
        return loss, (squared_error / v[0].numel()).mean().detach()
    return loss

def imf_loss(
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
    return_raw_mse=False,
):
    """Improved MeanFlow (iMF) velocity loss for a noise-to-data path.

    ``model`` predicts the average velocity u(z_r, r, t).  The iMF objective
    trains the re-parameterized instantaneous velocity

        V = u - (t-r) D_r u,

    against the fixed flow-matching target x_1 - x_0.  ``D_r u`` is evaluated
    with the boundary estimate u(z_r, r, r) and detached before the loss.  This
    is a first-order velocity regression; allowing gradients through the JVP
    would turn it into an expensive, unstable second-order objective.
    """
    device = x_1.device
    b = x_1.shape[0]

    x_0 = torch.randn_like(x_1)
 
    r, t = sample_ordered_times(
        b, device, time_sampler, logit_normal_mean, logit_normal_std
    )
 
    force_eq = torch.rand(b, device=device) < p_rt
    r = torch.where(force_eq, t, r)
 
    r_ = r.reshape(-1, 1, 1, 1)
    z_r = (1 - r_) * x_0 + r_ * x_1
    v_star = x_1 - x_0

    train_labels, pure_null_labels = _prepare_labels(labels, model, p_uncond, device)
    w, w_reshaped = _sample_w(labels, w_min, w_max, device)
    f = _make_cfg_fn(model, train_labels, pure_null_labels, w, w_reshaped)

    v_theta = f(z_r, r, r).detach()

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
 
    # u(z_r, r, r) satisfies the boundary condition u=v.  It supplies the
    # direction for the JVP while samples with r=t directly train it.
    v_theta = f(z_r, r, r).detach()
 
    primals  = (z_r, r, t)
    tangents = (v_theta, torch.ones_like(r), torch.zeros_like(t))
    model_output, du_dr = torch.func.jvp(f, primals, tangents)
 
    t_minus_r = (t - r).reshape(-1, 1, 1, 1)
    V_theta = improved_velocity_from_mean_velocity(
        model_output, du_dr.detach(), t_minus_r
    )

    # iMF uses adaptive loss weighting, not a Huber/clamped JVP surrogate.
    # The denominator is detached so it rescales per-example gradients without
    # changing the regression optimum.  Power 1 is the paper's default.
    squared_error = (V_theta - v_star).square().flatten(1).sum(dim=1)
    adaptive_weight = (squared_error.detach() + adaptive_loss_eps).pow(
        adaptive_loss_power
    )
    loss = (squared_error / adaptive_weight).mean()
    if return_raw_mse:
        # With adaptive_loss_power=1 the optimized loss is approximately one
        # by construction, so callers must log the unweighted MSE instead.
        return loss, (squared_error / v_star[0].numel()).mean().detach()
    return loss

    du_dr, du_dr_max = _clean_du_dr(du_dr, clamp_val)

    t_minus_r = (t - r).reshape(-1, 1, 1, 1)
    V_theta = model_output + t_minus_r * du_dr

    loss = F.huber_loss(V_theta, v_star)
    return loss, du_dr_max