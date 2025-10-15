from __future__ import annotations
from typing import Callable as Callable, Optional, Literal
import numpy as np
import torch

from ..core.types import Tensor, TorchProcess, DriftFnTorch, DiffusionFnTorch, ScoreFnTorch, IntegrationMethod
from ..core.grid import TimeGrid
from ..core.utils import ensure_batch_torch, diag_noise_torch

def _g_square(x: Tensor, g_val: float | Tensor) -> Tensor:
    """Return g^2 with shape broadcastable to x (B,D)."""
    if isinstance(g_val, torch.Tensor):
        return (g_val.to(device=x.device, dtype=x.dtype).view(1, -1)) ** 2
    return torch.full_like(x, float(g_val) ** 2)

@torch.no_grad()
def euler_maruyama_step_torch(
    proc: TorchProcess,
    x: Tensor,
    t_tensor: Tensor,
    dt: float,
) -> Tensor:
    """One EM step for dX = f dt + g dW with diagonal g."""
    t_scalar = float(t_tensor[0, 0].item())
    f0 = proc.drift(x, t_tensor)                   # (B,D)
    g0 = proc.diffusion(t_scalar)                  # scalar or (D,)
    x = x + f0 * dt + diag_noise_torch(x, g0, np.sqrt(dt))
    return x

@torch.no_grad()
def heun_sde_step_torch(
    proc: TorchProcess,
    x: Tensor,
    t_tensor: Tensor,
    dt: float,
) -> Tensor:
    """Stochastic Heun (improved Euler) with shared ΔW."""
    t_scalar = float(t_tensor[0, 0].item())
    g0 = proc.diffusion(t_scalar)
    dW = diag_noise_torch(x, g0, np.sqrt(dt))            # shared dW
    f0 = proc.drift(x, t_tensor)
    x_pred = x + f0 * dt + dW
    f1 = proc.drift(x_pred, t_tensor + dt)
    x = x + 0.5 * (f0 + f1) * dt + dW
    return x

@torch.no_grad()
def simulate_sde_torch(
    proc: TorchProcess,
    x0: Tensor,
    grid: TimeGrid,
    *,
    method: IntegrationMethod = "heun",
) -> tuple[Tensor, Tensor]:
    """Simulate any TorchProcess on a TimeGrid (t0→t1). Returns (xs, ts).

    - method="em"  : Euler–Maruyama
    - method="heun": stochastic Heun (shared ΔW)
    - method="exact": exact method
    """
    x, was_scalar = ensure_batch_torch(x0)
    B, D = x.shape
    device, dtype = x.device, x.dtype

    ts = grid.times_torch(device=device, dtype=dtype)    # (N,)
    xs = torch.empty((ts.numel(), B, D), device=device, dtype=dtype)
    xs[0] = x
    t = ts[0].expand(B, 1).clone()

    for k in range(ts.numel() - 1):
        dt = float((ts[k + 1] - ts[k]).item())
        if method == "exact":
            mean, std = proc.transition_mean_std(x, dt)
            x = mean + torch.randn_like(x) * std
        elif method == "em":
            x = euler_maruyama_step_torch(proc, x, t, dt)
        else:
            x = heun_sde_step_torch(proc, x, t, dt)
        xs[k + 1] = x
        t = ts[k + 1].expand(B, 1)
    return (xs[:, 0, :] if was_scalar else xs), ts


@torch.no_grad()
def reverse_pc_sampler_torch(
    x_T: Tensor,
    process: TorchProcess,
    score_model: ScoreFnTorch,
    grid: TimeGrid,
    snr: float = 0.15,
    eps_corrector: Optional[float] = None,
) -> Tensor:
    """Reverse-time Predictor–Corrector on a TimeGrid (t1 → t0)."""
    x, was_scalar = ensure_batch_torch(x_T)
    B, D = x.shape
    device, dtype = x.device, x.dtype
    f_fn = process.drift
    g_fn = process.diffusion

    ts  = grid.times_torch(device=device, dtype=dtype)   # (N,)
    dts = ts[1:] - ts[:-1]                               # (N-1,)

    t_tensor = ts[-1].expand(B, 1).clone()
    t_scalar = float(ts[-1].item())

    for k in range(dts.numel() - 1, -1, -1):
        dt = -float(dts[k].item())  # negative step
        g_val = g_fn(t_scalar)

        # Predictor (reverse EM)
        g_sq = _g_square(x, g_val)
        drift = f_fn(x, t_tensor) - g_sq * score_model(x, t_tensor)
        x = x + drift * dt + diag_noise_torch(x, g_val, np.sqrt(-dt))

        # Corrector (one Langevin step)
        s = score_model(x, t_tensor)
        g_for_eps = float(g_val.mean().item()) if isinstance(g_val, torch.Tensor) else float(g_val)
        eps = (snr * g_for_eps) ** 2 if eps_corrector is None else float(eps_corrector)
        x = x + eps * s + np.sqrt(2.0 * eps) * torch.randn_like(x)

        # Move to left node of the segment
        t_tensor = ts[k].expand(B, 1)
        t_scalar = float(ts[k].item())

    return x.squeeze(0) if was_scalar else x


def _pf_drift(
    x: Tensor, t_tensor: Tensor, t_scalar: float,
    f_fn: DriftFnTorch, g_fn: DiffusionFnTorch, score_model: ScoreFnTorch
) -> Tensor:
    g_val = g_fn(t_scalar)
    g2 = _g_square(x, g_val)
    return f_fn(x, t_tensor) - 0.5 * g2 * score_model(x, t_tensor)


def heun_step_pf_ode_torch(
    x: Tensor, t_tensor: Tensor, dt: float,
    f_fn: DriftFnTorch, g_fn: DiffusionFnTorch, score_model: ScoreFnTorch
) -> Tensor:
    t_scalar = float(t_tensor[0, 0].item())
    d1 = _pf_drift(x, t_tensor, t_scalar, f_fn, g_fn, score_model)
    x_pred = x + dt * d1
    t2_tensor = t_tensor + dt
    t2_scalar = t_scalar + dt
    d2 = _pf_drift(x_pred, t2_tensor, t2_scalar, f_fn, g_fn, score_model)
    return x + 0.5 * dt * (d1 + d2)


@torch.no_grad()
def prob_flow_sampler_torch(
    x_T: Tensor,
    score_model: ScoreFnTorch,
    f_fn: DriftFnTorch,
    g_fn: DiffusionFnTorch,
    grid: TimeGrid,
) -> Tensor:
    """Deterministic PF-ODE sampler on a TimeGrid (integrate t1 → t0)."""
    x, was_scalar = ensure_batch_torch(x_T)
    B = x.shape[0]
    device, dtype = x.device, x.dtype

    ts  = grid.times_torch(device=device, dtype=dtype)
    dts = ts[1:] - ts[:-1]

    t_tensor = ts[-1].expand(B, 1).clone()
    for k in range(dts.numel() - 1, -1, -1):
        dt = -float(dts[k].item())
        x = heun_step_pf_ode_torch(x, t_tensor, dt, f_fn, g_fn, score_model)
        t_tensor = ts[k].expand(B, 1)
    return x.squeeze(0) if was_scalar else x


def standard_normal_logp(x: Tensor, sigma: float = 1.0) -> Tensor:
    x_b, was_scalar = ensure_batch_torch(x)
    B, D = x_b.shape
    const = -0.5 * D * np.log(2.0 * np.pi * (sigma ** 2))
    quad  = -(x_b.pow(2).sum(dim=1)) / (2.0 * (sigma ** 2))
    out = x_b.new_full((B,), const) + quad
    return out.squeeze(0) if was_scalar else out


@torch.no_grad()
def pf_logp_from_x0(
    x0: Tensor,
    process: TorchProcess,
    score_model: ScoreFnTorch,
    prior_std: float,
    grid: TimeGrid,
    n_probe: int = 1,
) -> Tensor:
    """Estimate log p0(x0): integrate PF-ODE forward (t0→t1) along `grid`, add prior at t1."""
    x, was_scalar = ensure_batch_torch(x0)
    B = x.shape[0]
    device, dtype = x.device, x.dtype

    ts  = grid.times_torch(device=device, dtype=dtype)
    dts = ts[1:] - ts[:-1]

    t_tensor = ts[0].expand(B, 1).clone()
    logp_acc = torch.zeros(B, device=device, dtype=dtype)
    score_model = score_model.to(device=device, dtype=dtype)

    # Hutchinson divergence
    for k in range(dts.numel()):
        dt = float(dts[k].item())
        with torch.enable_grad():
            xx = x.detach().requires_grad_(True)
            acc = torch.zeros(B, device=x.device, dtype=x.dtype)
            for _ in range(n_probe):
                v = torch.randn_like(xx)
                t_scalar = float(t_tensor[0, 0].item())
                fpf = _pf_drift(xx, t_tensor, t_scalar, process.drift, process.diffusion, score_model)
                inner = (fpf * v).sum()
                (jtv,) = torch.autograd.grad(inner, xx, create_graph=False)
                acc = acc + (jtv * v).sum(dim=1)
            div = acc / float(n_probe)
        logp_acc = logp_acc - div * dt

        x = heun_step_pf_ode_torch(x, t_tensor, dt, process.drift, process.diffusion, score_model)
        t_tensor = ts[k + 1].expand(B, 1)

    logp_T = standard_normal_logp(x, sigma=prior_std)
    out = logp_T + logp_acc
    return out.squeeze(0) if was_scalar else out
