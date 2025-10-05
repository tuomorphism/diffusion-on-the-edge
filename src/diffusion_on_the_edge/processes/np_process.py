from __future__ import annotations
from dataclasses import dataclass
import numpy as np

from ..core.types import Array, NumpyProcess


@dataclass(frozen=True)
class OUParamsNP:
    dim: int = 1
    theta: float | Array = 0.5
    mu: float | Array = 0.0
    sigma: float | Array = 1.0

class OUNumpy(NumpyProcess):
    def __init__(self, params: OUParamsNP):
        self.dim = params.dim

        # Explicitly broadcast the parameters to suit the dimension of the process.
        self.theta = np.broadcast_to(params.theta, (self.dim,))
        self.mu = np.broadcast_to(params.mu, (self.dim,))
        self.sigma = np.broadcast_to(params.sigma, (self.dim,))

        # Simple assertions
        if np.any(self.theta <= 0): raise ValueError("theta must be > 0")
        if np.any(self.sigma < 0):  raise ValueError("sigma must be >= 0")

    def drift(self, x: Array, _: float) -> Array:
        x = np.asarray(x, float)
        return self.theta * (self.mu - x[..., -self.dim:])

    def diffusion(self, _: float) -> Array:
        return self.sigma

    def transition_mean_std(self, x: Array, dt: float) -> tuple[Array, Array]:
        e = np.exp(-self.theta * dt)
        mean = self.mu + (x[..., -self.dim:] - self.mu) * e
        var  = (self.sigma**2) * (1.0 - e**2) / (2.0 * self.theta)
        std  = np.sqrt(var)
        return mean, std
