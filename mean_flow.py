import torch
import torch.nn.functional as F
from torch.nn.attention import sdpa_kernel, SDPBackend


def _sample_r_t(b, device, p_rt):
    a = torch.rand(b, device=device)
    c = torch.rand(b, device=device)
    t = torch.maximum(a, c)
    r = torch.minimum(a, c)
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


def _safe_jvp(f, primals, tangents, use_math_kernel=False):
    """Runs jvp in fp32 regardless of outer autocast, with optional MATH-only SDPA."""
    with torch.autocast(device_type="cuda", enabled=False):
        primals_fp32 = tuple(p.float() for p in primals)
        tangents_fp32 = tuple(t_.float() for t_ in tangents)
        if use_math_kernel:
            with sdpa_kernel(SDPBackend.MATH):
                out, tangent_out = torch.func.jvp(f, primals_fp32, tangents_fp32)
        else:
            out, tangent_out = torch.func.jvp(f, primals_fp32, tangents_fp32)
    return out.float(), tangent_out


def _clean_du_dr(du_dr, clamp_val=100.0):
    #du_dr = torch.nan_to_num(du_dr, nan=0.0, posinf=clamp_val, neginf=-clamp_val)
    du_dr = du_dr.detach()
    du_dr_max = du_dr.abs().max().item()
    du_dr = torch.clamp(du_dr, min=-clamp_val, max=clamp_val)
    return du_dr, du_dr_max

def adaptive_l2_loss(pred, target, p=0.75, c=1e-3):
    error = pred - target
    error_sq = error.pow(2).flatten(1).mean(dim=1)  # per-sample scalar MSE
    weight = 1.0 / (error_sq.detach() + c).pow(p)   # detach = don't backprop through the weight itself
    loss = (weight * error_sq).mean()
    return loss

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
    u, du_dr = _safe_jvp(f, primals, tangents)

    du_dr, du_dr_max = _clean_du_dr(du_dr, clamp_val)

    t_minus_r = (t - r).reshape(-1, 1, 1, 1)
    u_target = v + t_minus_r * du_dr

    loss = adaptive_l2_loss(u, u_target.detach(), p=0.75, c=1e-3)
    return loss, du_dr_max


def imf_loss(model, x_1, labels, p_rt=0.5, p_uncond=0.1, w_min=1.0, w_max=5.0, clamp_val=100.0):
    device = x_1.device
    b = x_1.shape[0]

    x_0 = torch.randn_like(x_1)
    r, t = _sample_r_t(b, device, p_rt)

    r_ = r.reshape(-1, 1, 1, 1)
    z_r = (1 - r_) * x_0 + r_ * x_1
    v_star = x_1 - x_0

    train_labels, pure_null_labels = _prepare_labels(labels, model, p_uncond, device)
    w, w_reshaped = _sample_w(labels, w_min, w_max, device)
    f = _make_cfg_fn(model, train_labels, pure_null_labels, w, w_reshaped)

    v_theta = f(z_r, r, r).detach()

    primals = (z_r, r, t)
    tangents = (v_theta, torch.ones_like(r), torch.zeros_like(t))
    model_output, du_dr = _safe_jvp(f, primals, tangents, use_math_kernel=True)

    du_dr, du_dr_max = _clean_du_dr(du_dr, clamp_val)

    t_minus_r = (t - r).reshape(-1, 1, 1, 1)
    V_theta = model_output + t_minus_r * du_dr

    loss = adaptive_l2_loss(V_theta, v_star, p=0.75, c=1e-3)
    return loss, du_dr_max