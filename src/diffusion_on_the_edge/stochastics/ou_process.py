from typing import Tuple
from dataclasses import dataclass
import numpy as np

from diffusion_on_the_edge.stochastics.sde_np import improved_euler_sde_np

Array = np.ndarray

@dataclass 
class Process:
    T_0: float = 0.0
    T_1: float = 1.0
    temporal_resolution: int = 100
    dimension: int = 1
    spatial_resolution: int = 100

    def grid(self) -> Array:
        return np.linspace(start = self.T_0, stop = self.T_1, num = self.temporal_resolution)

@dataclass
class OUParameters:
    lambda_coeff: float = 0.5
    sigma_coeff: float = 1.0

def generate_ou_trajectory(x0: Array, parameters: OUParameters, process: Process) -> Tuple[Array, Array]:
    def _f(x: Array, t: float):
        return -parameters.lambda_coeff * np.ones(process.dimension)
    g = lambda t: parameters.sigma_coeff
    grid = TimeGrid(t0 = process.T_0, t1 =process.T_1, N = temporal_resolution)
    values = improved_euler_sde_np(x0, grid, _f, g)
    trajectory = np.asarray([v[0] for v in values])
    timesteps = np.asarray([v[1] for v in values])
    return (trajectory, timesteps)



def generate_ou_density(x0: Array, x: Array, process: OUParameters, temporal_resolution: int = 100) -> Tuple[Array, Array]:
    """
    Generates an density value for each spatial point in x and for each timestep in the OU process.
    """
    t_range = np.linspace(process.T_0, process.T_1, num = temporal_resolution)
    means = np.exp(-process.lambda_coeff * t_range) * x0
    variances = (process.sigma_coeff ** 2) / (2 * process.lambda_coeff) * (1 - np.exp(-2 * process.lambda_coeff * t_range)) * np.ones(process.dimension)

