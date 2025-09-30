# diffusion_on_the_edge/processes/ou.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
from numpy.typing import NDArray

Array = NDArray[np.floating]




# ---------------------------------------------------------------------
# OU parameters (immutable)
# ---------------------------------------------------------------------
@dataclass(frozen=True)
class OUParams:
    """Parameters for an Ornstein–Uhlenbeck process with independent coordinates.

    The SDE is
        dX_t = θ * (μ - X_t) dt + σ dW_t,

    applied componentwise.

    Parameters
    ----------
    theta : float | Array
        Mean reversion rate(s) θ > 0. Broadcastable to shape (dim,).
    mu : float | Array
        Long-run mean(s) μ. Broadcastable to (dim,).
    sigma : float | Array
        Volatility σ >= 0. Broadcastable to (dim,).
    dim : int
        Dimension of the state.
    """

    theta: float | Array = 0.5
    mu: float | Array = 0.0
    sigma: float | Array = 1.0
    dim: int = 1


# ---------------------------------------------------------------------
# OU process with exact transition
# ---------------------------------------------------------------------
class OUProcess:
    """Ornstein–Uhlenbeck process with exact Gaussian transitions.

    Provides drift/diffusion for Euler–Maruyama compatibility and an
    exact `step`/`simulate` based on closed-form transitions.
    """

    def __init__(self, params: OUParams):
        self.params = params
        self.dim = int(params.dim)

        # Broadcast scalars to vectors of length dim for fast vectorized math
        self.theta = np.broadcast_to(np.asarray(params.theta, dtype=float), (self.dim,))
        self.mu = np.broadcast_to(np.asarray(params.mu, dtype=float), (self.dim,))
        self.sigma = np.broadcast_to(np.asarray(params.sigma, dtype=float), (self.dim,))

        # Basic checks
        if np.any(self.theta <= 0):
            raise ValueError("All theta must be > 0.")
        if np.any(self.sigma < 0):
            raise ValueError("All sigma must be >= 0.")

    # ---- SDE pieces (for compatibility with generic SDE interfaces) ----
    def drift(self, x: Array, t: float) -> Array:
        x = np.asarray(x, dtype=float).reshape(self.dim)
        return self.theta * (self.mu - x)

    def diffusion(self, x: Array, t: float) -> Array:
        # Diagonal (independent) noise
        return self.sigma

    # ---- Exact transition for one step ----
    def transition_mean_var(self, x: Array, dt: float) -> Tuple[Array, Array]:
        """Return conditional mean and variance diag of X_{t+dt} | X_t = x.

        Parameters
        ----------
        x : Array
            Current state (dim,).
        dt : float
            Positive step size.

        Returns
        -------
        mean : Array
            Mean of next state (dim,).
        var : Array
            Variance (diagonal) of next state (dim,).
        """
        if dt < 0:
            raise ValueError("dt must be >= 0")
        x = np.asarray(x, dtype=float).reshape(self.dim)

        e = np.exp(-self.theta * dt)  # (dim,)
        mean = self.mu + (x - self.mu) * e
        # Avoid 0/0 when theta -> 0 (not our case since we enforce > 0)
        var = (self.sigma**2) * (1.0 - e**2) / (2.0 * self.theta)
        return mean, var

    def step(self, x: Array, t: float, dt: float, rng: Optional[np.random.Generator] = None) -> Array:
        """Exact one-step update using the Gaussian transition."""
        if rng is None:
            rng = np.random.default_rng()
        mean, var = self.transition_mean_var(x, dt)
        z = rng.normal(size=self.dim)
        return mean + np.sqrt(var) * z

    def simulate(self, x0: Array, grid: TimeGrid, rng: Optional[np.random.Generator] = None) -> Tuple[Array, Array]:
        """Simulate a path on `grid` using exact OU transitions.

        Parameters
        ----------
        x0 : Array
            Initial state (dim,) at time grid.t0.
        grid : TimeGrid
            Time discretization.
        rng : np.random.Generator, optional
            Random generator.

        Returns
        -------
        xs : Array
            Array of shape (N, dim) with states at each grid point.
        ts : Array
            Array of shape (N,) with the time stamps.
        """
        if rng is None:
            rng = np.random.default_rng()
        ts = grid.times()
        N = ts.shape[0]

        xs = np.empty((N, self.dim), dtype=float)
        x = np.asarray(x0, dtype=float).reshape(self.dim)
        xs[0] = x

        for i in range(1, N):
            dt = float(ts[i] - ts[i - 1])
            x = self.step(x, float(ts[i - 1]), dt, rng)
            xs[i] = x
        return xs, ts

    # ---- Analytic marginal (given deterministic x0 at t=0) ----
    def marginal_mean_var(self, x0: Array, t: float) -> Tuple[Array, Array]:
        """Mean/variance of X_t given X_0 = x0 (componentwise)."""
        e = np.exp(-self.theta * t)
        mean = self.mu + (np.asarray(x0, float).reshape(self.dim) - self.mu) * e
        var = (self.sigma**2) * (1.0 - e**2) / (2.0 * self.theta)
        return mean, var

    def marginal_density(self, x: Array, t: float, x0: Array) -> float:
        """Product Gaussian density p(X_t = x | X_0 = x0) for independent components.

        For dim==1 this is the standard univariate OU density.
        """
        mean, var = self.marginal_mean_var(x0, t)
        x = np.asarray(x, float).reshape(self.dim)
        # Product of independent normals
        log_norm = -0.5 * np.sum(np.log(2.0 * np.pi * var))
        quad = -0.5 * np.sum(((x - mean) ** 2) / var)
        return float(np.exp(log_norm + quad))


# ---------------------------------------------------------------------
# Convenience functions (migration-friendly)
# ---------------------------------------------------------------------
def generate_ou_trajectory(
    x0: Array,
    parameters: OUParams,
    grid: TimeGrid,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[Array, Array]:
    """Simulate an OU trajectory (compatibility shim for the old API).

    Returns
    -------
    trajectory : Array
        Shape (N, dim).
    timesteps : Array
        Shape (N,).
    """
    ou = OUProcess(parameters)
    xs, ts = ou.simulate(x0=x0, grid=grid, rng=rng)
    return xs, ts


def generate_ou_density(
    x: Array,
    parameters: OUParams,
    grid: TimeGrid,
    x0: Array | float | None = None,
) -> Array:
    """Compute OU marginal density over a spatial grid of points and times.

    Parameters
    ----------
    x : Array
        Spatial points. For 1D, shape (M,). For d-D independent components, shape (M, dim).
    parameters : OUParams
        OU parameters.
    grid : TimeGrid
        Time discretization t in [t0, t1].
    x0 : Array | float | None
        Deterministic initial condition X_{t0}. If None, uses zeros.

    Returns
    -------
    density : Array
        If x is (M,) and dim==1, returns shape (N, M) with p(X_t = x_j) for each time.
        If x is (M, dim), returns shape (N, M) with product Gaussian densities.

    Notes
    -----
    This uses the closed-form OU marginal with independent coordinates.
    """
    ou = OUProcess(parameters)
    ts = grid.times()
    dim = ou.dim

    # Normalize x grid shape
    x_arr = np.asarray(x, float)
    if x_arr.ndim == 1 and dim == 1:
        x_arr = x_arr[:, None]  # (M, 1)
    elif x_arr.ndim == 1 and dim > 1:
        raise ValueError("For dim > 1, provide x with shape (M, dim).")
    elif x_arr.shape[1] != dim:
        raise ValueError(f"x must have shape (M, {dim}).")

    M = x_arr.shape[0]
    x0_arr = np.zeros((dim,), dtype=float) if x0 is None else np.asarray(x0, float).reshape(dim)

    out = np.empty((ts.shape[0], M), dtype=float)
    for i, t in enumerate(ts):
        mean, var = ou.marginal_mean_var(x0_arr, float(t - ts[0]))  # relative to t0
        # product Gaussian density at all M points
        log_norm = -0.5 * np.sum(np.log(2.0 * np.pi * var))
        quad = -0.5 * np.sum(((x_arr - mean) ** 2) / var, axis=1)
        out[i] = np.exp(log_norm + quad)
    return out
