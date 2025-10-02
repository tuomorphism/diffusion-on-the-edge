# diffusion_on_the_edge/sde/np_backend.py
from __future__ import annotations

from typing import Callable, Optional, Tuple
import numpy as np
from numpy.typing import NDArray

from ..core.grid import TimeGrid
from ..core.utils import ensure_batch_np, diag_noise_np
from ..core.types import NumpyProcess, Array, DriftFnNP, DiffusionFnNP, ScoreFnNP


__all__ = [
    "improved_euler_sde_np",
    "reverse_pc_sampler_np",
    "prob_flow_ode_step_np",
    "prob_flow_sampler_np",
    "simulate_sde_np",
]


def _g_square_np(x_like: Array, g_val: Array | float) -> Array:
    """Return g(t)^2 broadcastable to x_like (..., D). Accepts scalar or (D,)."""
    if np.ndim(g_val) == 0:
        return np.full_like(x_like, float(g_val) ** 2, dtype=float)
    g_vec = np.asarray(g_val, dtype=float).reshape((1,) * (x_like.ndim - 1) + (-1,))
    return g_vec**2


def improved_euler_sde_np(
    x0: Array,
    grid: TimeGrid,
    f: DriftFnNP,
    g: DiffusionFnNP,
) -> Tuple[Array, Array]:
    """Stochastic Heun / improved Euler for Itô SDE with diagonal diffusion.

    Uses a single shared Wiener increment ΔW for predictor and corrector:
        X̂ = X_k + f(X_k, t_k) Δt + g(t_k) ΔW
        X_{k+1} = X_k + 0.5[ f(X_k, t_k) + f(X̂, t_{k+1}) ] Δt + g(t_k) ΔW

    Supports x0 with shape (D,) or (B, D). Returns (xs, ts).
    """
    ts = grid.times()
    N = ts.size

    x, was_scalar = ensure_batch_np(np.asarray(x0, dtype=float))
    B, D = x.shape

    xs = np.empty((N, B, D), dtype=float)
    xs[0] = x

    for k in range(1, N):
        t_prev = float(ts[k - 1])
        t_curr = float(ts[k])
        dt = t_curr - t_prev
        if dt <= 0:
            raise ValueError("TimeGrid must be strictly increasing.")
        sqrt_dt = np.sqrt(dt)

        g_prev = g(t_prev)  # scalar or (D,)
        dW = diag_noise_np((B, D), g_prev, sqrt_dt)  # shared ΔW

        f_prev = f(xs[k - 1], t_prev)            # (B, D)
        x_pred = xs[k - 1] + dt * f_prev + dW
        f_pred = f(x_pred, t_curr)               # (B, D)
        f_avg = 0.5 * (f_prev + f_pred)

        x_next = xs[k - 1] + dt * f_avg + dW
        xs[k] = x_next

    return (xs[:, 0, :] if was_scalar else xs), ts


def reverse_pc_sampler_np(
    x_T: Array,
    score: ScoreFnNP,
    f: DriftFnNP,
    g: DiffusionFnNP,
    T: float = 1.0,
    N: int = 1000,
    snr: float = 0.15,
    eps_corrector: Optional[float] = None,
) -> Array:
    """Reverse-time Predictor–Corrector sampler in NumPy.

    Predictor (reverse EM):
        dX = [ f(X,t) - g(t)^2 * score(X,t) ] dt + g(t) dW, with dt < 0.
    Corrector (one Langevin step):
        X ← X + ε score + sqrt(2ε) ξ, with ε = (snr * g(t))^2 by default.
    """
    x = np.asarray(x_T, dtype=float)
    scalar = x.ndim == 1
    if scalar:
        x = x[None, :]
    B, D = x.shape

    dt_pos = T / N
    dt = -dt_pos
    t = float(T)

    for _ in range(N):
        gval = g(t)                                  # scalar or (D,)
        drift = f(x, t) - _g_square_np(x, gval) * score(x, t)
        x = x + drift * dt + diag_noise_np((B, D), gval, np.sqrt(-dt))

        # Corrector step
        s = score(x, t)
        g_for_eps = float(np.mean(gval)) if np.ndim(gval) else float(gval)
        eps = (snr * g_for_eps) ** 2 if eps_corrector is None else float(eps_corrector)
        x = x + eps * s + np.sqrt(2.0 * eps) * np.random.normal(size=x.shape)

        t += dt

    return x.squeeze() if scalar else x


def prob_flow_ode_step_np(
    x: Array, t: float, dt: float, f: DriftFnNP, g: DiffusionFnNP, score: ScoreFnNP
) -> Array:
    """One Heun step for the probability-flow ODE:
        dX/dt = f(X,t) - 0.5 * g(t)^2 * score(X,t)
    """
    def drift(xx: Array, tt: float) -> Array:
        gval = g(tt)
        if np.ndim(gval) == 0:
            return f(xx, tt) - 0.5 * (float(gval) ** 2) * score(xx, tt)
        g_vec2 = np.asarray(gval, dtype=float) ** 2  # (D,)
        return f(xx, tt) - 0.5 * g_vec2 * score(xx, tt)

    x_pred = x + dt * drift(x, t)
    return x + 0.5 * dt * (drift(x, t) + drift(x_pred, t + dt))


def prob_flow_sampler_np(
    x_T: Array,
    f: DriftFnNP,
    g: DiffusionFnNP,
    score: ScoreFnNP,
    T: float = 1.0,
    N: int = 1000,
) -> Array:
    """Deterministic probability-flow ODE sampler (NumPy)."""
    x = np.asarray(x_T, dtype=float)
    scalar = x.ndim == 1
    if scalar:
        x = x[None, :]

    dt = -T / N
    t = float(T)
    for _ in range(N):
        x = prob_flow_ode_step_np(x, t, dt, f, g, score)
        t += dt

    return x.squeeze() if scalar else x


def simulate_sde_np(
    proc: NumpyProcess,
    x0: Array,
    grid: TimeGrid,
    *,
    method: str = "heun",   # "em" | "heun" | "exact_if_available"
) -> tuple[Array, Array]:
    """Simulate a NumpyProcess on a TimeGrid using chosen method."""
    ts = grid.times()
    x, was_scalar = ensure_batch_np(np.asarray(x0, dtype=float))
    B, D = x.shape
    xs = np.empty((ts.size, B, D), dtype=float)
    xs[0] = x

    use_exact = method == "exact_if_available" and hasattr(proc, "transition_mean_std")

    for k in range(ts.size - 1):
        t0 = float(ts[k])
        dt = float(ts[k + 1] - ts[k])
        sqrt_dt = np.sqrt(dt)

        if use_exact:
            m, s = proc.transition_mean_std(x, dt)
            x = m + np.random.normal(size=x.shape) * s
        elif method == "em":
            f0 = proc.drift(x, t0)
            g0 = proc.diffusion(t0)
            x = x + f0 * dt + diag_noise_np((B, D), g0, sqrt_dt)
        else:
            f0 = proc.drift(x, t0)
            g0 = proc.diffusion(t0)
            dW = diag_noise_np((B, D), g0, sqrt_dt)
            x_pred = x + f0 * dt + dW
            f1 = proc.drift(x_pred, float(ts[k + 1]))
            x = x + 0.5 * (f0 + f1) * dt + dW

        xs[k + 1] = x

    return (xs[:, 0, :] if was_scalar else xs), ts
