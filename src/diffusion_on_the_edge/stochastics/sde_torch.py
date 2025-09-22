from typing import Callable as _Callable, Tuple, Optional
import torch
import numpy as np

Tensor = torch.Tensor
DriftFnTorch = _Callable[[Tensor, Tensor], Tensor]
DiffusionFnTorch = _Callable[[float], float]
ScoreFnTorch = _Callable[[Tensor, Tensor], Tensor]


@torch.no_grad()
def reverse_pc_sampler_torch(
    x_T: Tensor,
    score_model: ScoreFnTorch,
    f_fn: DriftFnTorch,
    g_fn: DiffusionFnTorch,
    T: float = 1.0,
    N: int = 1000,
    snr: float = 0.15,
    eps_corrector: Optional[float] = None,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
) -> Tensor:
    device = device or x_T.device
    dtype = dtype or x_T.dtype
    x = x_T.to(device=device, dtype=dtype)
    dt_pos = T / N
    dt = -dt_pos
    t = torch.full((x.shape[0], 1), T, device=device, dtype=dtype)

    for _ in range(N):
        t_scalar = float(t[0, 0].item())
        g = torch.as_tensor(g_fn(t_scalar), device=device, dtype=dtype)
        drift = f_fn(x, t) - (g**2) * score_model(x, t)
        x = x + drift * dt + g * torch.sqrt(torch.tensor(abs(dt), device=device, dtype=dtype)) * torch.randn_like(x)

        s = score_model(x, t)
        eps = (snr * g) ** 2 if eps_corrector is None else torch.as_tensor(eps_corrector, device=device, dtype=dtype)
        x = x + eps * s + torch.sqrt(2.0 * eps) * torch.randn_like(x)

        t = t - dt_pos
    return x


def _pf_drift_torch(x: Tensor, t: Tensor, f_fn: DriftFnTorch, g_fn: DiffusionFnTorch, score_model: ScoreFnTorch) -> Tensor:
    """Probability-flow ODE drift: f - 0.5 g^2 score."""
    # t used as scalar in g_fn, but keep batched t for f/score
    t_scalar = float(t[0, 0].item()) if t.ndim == 2 else float(t.item())
    g2 = g_fn(t_scalar) ** 2
    return f_fn(x, t) - 0.5 * g2 * score_model(x, t)


def heun_step_pf_ode_torch(x: Tensor, t: Tensor, dt: float, f_fn: DriftFnTorch, g_fn: DiffusionFnTorch, score_model: ScoreFnTorch) -> Tensor:
    """One Heun (predictor–corrector) step for PF-ODE."""
    d1 = _pf_drift_torch(x, t, f_fn, g_fn, score_model)
    x_pred = x + dt * d1
    d2 = _pf_drift_torch(x_pred, t + dt, f_fn, g_fn, score_model)
    return x + 0.5 * dt * (d1 + d2)


@torch.no_grad()
def prob_flow_sampler_torch(
    x_T: Tensor,
    score_model: ScoreFnTorch,
    f_fn: DriftFnTorch,
    g_fn: DiffusionFnTorch,
    T: float = 1.0,
    N: int = 1000,
) -> Tensor:
    """Deterministic probability-flow ODE sampler (Torch)."""
    x = x_T
    dt = -T / N
    t = torch.full((x.shape[0], 1), T, device=x.device, dtype=x.dtype)
    for _ in range(N):
        x = heun_step_pf_ode_torch(x, t, dt, f_fn, g_fn, score_model)
        t = t + dt
    return x

def standard_normal_logp(x: Tensor, sigma: float = 1.0) -> Tensor:
    """Log-density of N(0, sigma^2 I) per sample."""
    D = x[0].numel()
    const = -0.5 * D * np.log(2 * np.pi * (sigma ** 2))
    quad = - (x.pow(2).sum(dim=tuple(range(1, x.ndim))) ) / (2 * sigma ** 2)
    return x.new_tensor(const) + quad


def sample_ou_exact_np(x0: np.ndarray, t: float, lam: float, sigma: float, rng: Optional[np.random.Generator] = None) -> np.ndarray:
    """Exact OU sample in NumPy (kept for parity).
    dX = -lam X dt + sigma dW.
    """
    rng = rng or np.random.default_rng()
    decay = np.exp(-lam * t)
    variance = (sigma**2 / (2 * lam)) * (1 - np.exp(-2 * lam * t))
    return decay * x0 + np.sqrt(variance) * rng.standard_normal(size=x0.shape)

def _divergence_hutchinson_pf(
    x: Tensor, t: Tensor,
    f_fn: DriftFnTorch, g_fn: DiffusionFnTorch, score_model: ScoreFnTorch,
    n_probe: int = 1
) -> Tensor:
    """Estimate div f_pf(x,t) with Hutchinson; grads enabled only here."""
    with torch.enable_grad():
        xx = x.detach().requires_grad_(True)
        acc = 0.0
        for _ in range(n_probe):
            v = torch.randn_like(xx)
            fpf = _pf_drift_torch(xx, t, f_fn, g_fn, score_model)
            inner = (fpf * v).sum()
            (jtv,) = torch.autograd.grad(inner, xx, create_graph=False)
            acc = acc + (jtv * v).sum(dim=tuple(range(1, jtv.ndim)))
        return acc / float(n_probe)  # shape (B,)


@torch.no_grad()
def pf_logp_from_x0(
    x0: Tensor,
    score_model: ScoreFnTorch,
    f_fn: DriftFnTorch,
    g_fn: DiffusionFnTorch,
    prior_variance: float,
    T: float = 1.0,
    N: int = 1000,
    n_probe: int = 1,
) -> Tensor:
    """
    Compute log p0(x0) via PF-ODE forward (0->T) + ICOV using your helpers.
    """
    x = x0.clone()
    device, dtype = x.device, x.dtype
    t = torch.zeros((x.shape[0], 1), device=device, dtype=dtype)
    dt = T / N
    logp_acc = torch.zeros(x.shape[0], device=device, dtype=dtype)

    for _ in range(N):
        div = _divergence_hutchinson_pf(x, t, f_fn, g_fn, score_model, n_probe=n_probe)
        logp_acc -= div * dt
        x = heun_step_pf_ode_torch(x, t, dt, f_fn, g_fn, score_model)
        t = t + dt

    # prior at T, using the OU closed form
    logp_T = standard_normal_logp(x, sigma=prior_variance)
    return logp_T + logp_acc  # == log p0(x0)
