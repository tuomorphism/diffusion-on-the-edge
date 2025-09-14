from typing import Callable as _Callable
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
    """Reverse-time Predictor (EM) + Corrector (Langevin) per level.

    Args mirror the NumPy version. score_model(x,t) returns ∇_x log p_t(x).
    """
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


def _divergence_hutchinson(x: Tensor, t: Tensor, drift_fn: DriftFnTorch) -> Tensor:
    """Estimate divergence wrt x via Hutchinson’s trace estimator.

    Returns per-sample divergence with a single probe vector v ~ N(0, I):
      div ≈ v^T J_drift(x) v
    where (J_drift v) is computed by autograd on (drift · v).
    """
    x = x.detach().requires_grad_(True)
    v = torch.randn_like(x)
    drift = drift_fn(x, t)
    # Compute (J^T v) by taking gradient of (drift · v) wrt x
    inner = (drift * v).sum()
    grad = torch.autograd.grad(inner, x, create_graph=True)[0]
    # Now take dot with v again to get v^T J v = (J^T v)·v
    div_est = (grad * v).sum(dim=tuple(range(1, grad.ndim)))
    return div_est  # shape (B,)


@torch.no_grad()
def prob_flow_density_and_sample_torch(
    x_T: Tensor,
    score_model: ScoreFnTorch,
    f_fn: DriftFnTorch,
    g_fn: DiffusionFnTorch,
    prior_logp_fn: _Callable[[Tensor], Tensor],
    T: float = 1.0,
    N: int = 1000,
) -> Tuple[Tensor, Tensor]:
    """Integrate PF-ODE backward (T→0) to get x_0 and log p_0(x_0).

    We track log-density via continuity equation along the PF-ODE flow:
      d log p / dt = - div_x( f_pf(x,t) ),  where f_pf = f - 0.5 g^2 score.

    Using Hutchinson to estimate divergence. With backward integration (dt<0):
      log p_0(x_0) ≈ log p_T(x_T) + \sum_k [ -div(f_pf(x_k,t_k)) * dt ].

    Args:
      x_T: samples at terminal time T from the prior.
      prior_logp_fn: callable giving log p_T(x_T) for the chosen prior.

    Returns:
      x0, logp0  (both tensors of shape (B, D) and (B,))
    """
    # We need autograd when computing divergence; keep graph for that part only.
    x = x_T.detach()
    dt = -T / N
    t = torch.full((x.shape[0], 1), T, device=x.device, dtype=x.dtype)

    logp = prior_logp_fn(x)  # log p_T(x_T)

    for _ in range(N):
        # Divergence at (x,t)
        div = _divergence_hutchinson(x, t, lambda xx, tt: _pf_drift_torch(xx, tt, f_fn, g_fn, score_model))
        logp = logp - div * dt  # accumulate with dt<0
        # Advance state with Heun step (no grad needed for state update)
        x = heun_step_pf_ode_torch(x, t, dt, f_fn, g_fn, score_model)
        t = t + dt
    return x, logp


# ------- Convenience Priors & OU exact sampler (torch) -------

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
