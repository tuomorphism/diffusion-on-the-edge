from dataclasses import dataclass
from typing import Callable, Tuple, Optional, List
import numpy as np

Array = np.ndarray

DriftFnNP = Callable[[Array, float], Array]
DiffusionFnNP = Callable[[float], float]
ScoreFnNP = Callable[[Array, float], Array]

def improved_euler_sde_np(
    x0: Array,
    grid: TimeGrid,
    f: DriftFnNP,
    g: DiffusionFnNP,
    rng: Optional[np.random.Generator] = None,
) -> List[Tuple[Array, float]]:
    """Stochastic Heun / improved Euler for Itô SDE:
    dX = f(X,t) dt + g(t) dW.

    Returns the full trajectory [(x_t, t), ...].
    """
    rng = rng or np.random.default_rng()
    ts = grid.grid()
    dt = (grid.t1 - grid.t0) / grid.N
    sqrt_dt = np.sqrt(dt)

    traj: List[Tuple[Array, float]] = [(np.array(x0, dtype=float), ts[0])]
    for k in range(1, ts.size):
        t_prev = ts[k-1]
        x_prev = traj[-1][0]

        g_prev = g(t_prev)
        noise = g_prev * rng.standard_normal(size=x_prev.shape)
        x_pred = x_prev + dt * f(x_prev, t_prev) + sqrt_dt * noise

        t_curr = ts[k]
        drift_pred = f(x_pred, t_curr)
        drift_avg = 0.5 * (f(x_prev, t_prev) + drift_pred)

        g_curr = g(t_curr)
        noise_next = g_curr * rng.standard_normal(size=x_prev.shape)
        x_next = x_prev + dt * drift_avg + sqrt_dt * noise_next
        traj.append((x_next, t_curr))
    return traj


def reverse_pc_sampler_np(
    x_T: Array,
    score: ScoreFnNP,
    f: DriftFnNP,
    g: DiffusionFnNP,
    T: float = 1.0,
    N: int = 1000,
    snr: float = 0.15,
    eps_corrector: Optional[float] = None,
    rng: Optional[np.random.Generator] = None,
) -> Array:
    """Reverse-time Predictor–Corrector in NumPy.

    Predictor: reverse-time EM for SDE
      dx = [f(x,t) - g(t)^2 * score(x,t)] dt + g(t) dW, with dt negative.
    Corrector: one Langevin step with step-size eps.

    x_T: batch of noisy latents at time T. Shape (B, D) or (D,).
    """
    rng = rng or np.random.default_rng()
    x = np.array(x_T, dtype=float)
    if x.ndim == 1:
        x = x[None, :]

    dt_pos = T / N
    dt = -dt_pos
    t = np.full((x.shape[0], 1), T, dtype=float)

    for _ in range(N):
        t_scalar = float(t[0, 0])
        gval = float(g(t_scalar))
        drift = f(x, t_scalar) - (gval**2) * score(x, t_scalar)
        x = x + drift * dt + gval * np.sqrt(abs(dt)) * rng.standard_normal(size=x.shape)

        s = score(x, t_scalar)
        eps = (snr * gval) ** 2 if eps_corrector is None else float(eps_corrector)
        x = x + eps * s + np.sqrt(2.0 * eps) * rng.standard_normal(size=x.shape)

        t = t - dt_pos
    return x.squeeze()


def prob_flow_ode_step_np(x: Array, t: float, dt: float, f: DriftFnNP, g: DiffusionFnNP, score: ScoreFnNP) -> Array:
    """One Heun step for probability-flow ODE: dX = [f - 0.5 g^2 score] dt.
    Useful for quick deterministic sampling when you don't need density.
    """
    drift = lambda xx, tt: f(xx, tt) - 0.5 * g(tt) ** 2 * score(xx, tt)
    x_pred = x + dt * drift(x, t)
    x_next = x + 0.5 * dt * (drift(x, t) + drift(x_pred, t + dt))
    return x_next


def prob_flow_sampler_np(
    x_T: Array,
    f: DriftFnNP,
    g: DiffusionFnNP,
    score: ScoreFnNP,
    T: float = 1.0,
    N: int = 1000,
) -> Array:
    """Deterministic PF-ODE sampler (NumPy, no density tracking)."""
    x = np.array(x_T, dtype=float)
    if x.ndim == 1:
        x = x[None, :]
    dt = -T / N
    t = float(T)
    for _ in range(N):
        x = prob_flow_ode_step_np(x, t, dt, f, g, score)
        t += dt
    return x.squeeze()