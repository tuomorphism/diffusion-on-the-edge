from __future__ import annotations
from typing import Protocol, Callable, Literal
from numpy.typing import NDArray
import torch
import numpy as np

# Array for numpy, Tensor for PyTorch.
Array = NDArray[np.floating]
Tensor = torch.Tensor

# Function signatures for drift and diffusion
DriftFnNP = Callable[[Array, float], Array]
DiffusionFnNP = Callable[[float], Array | float]
DriftFnTorch  = Callable[[Tensor, Tensor], Tensor]
ScoreFnTorch  = Callable[[Tensor, Tensor], Tensor]

# Additional type for score function, s(x, t)
ScoreFnNP = Callable[[Array, float], Array]
DiffusionFnTorch = Callable[[float], float | Tensor]

# Different stochastic integration methods
IntegrationMethod = Literal['em', 'heun', 'exact']

# Defining some process Protocols for implementation classes to use.
# These reflect the general diffusion process with drift and diffusion (noise) terms.
# Both numpy and PyTorch versions
class NumpyProcess(Protocol):
    dim: int

    # Drift function f(x, t)
    drift: DriftFnNP

    # Diffusion function g(t)
    diffusion: DiffusionFnNP

    # Analytical computations for the means and standard deviations of the process.
    # Starting from x, and computing the mean and std at time t.
    def transition_mean_std(self, x: Array, t: float) -> tuple[Array, Array]: ...

class TorchProcess(Protocol):
    # Same functions as NumpyProcess, just expressed as PyTorch tensor inputs and outputs where appropriate.

    dim: int
    drift: DriftFnTorch
    diffusion: DiffusionFnTorch

    def transition_mean_std(self, x: Tensor, t: Tensor) -> tuple[Tensor, Tensor]: ...
