from __future__ import annotations
from typing import Protocol
from numpy.typing import NDArray
import torch
import numpy as np

# Array for numpy, Tensor for PyTorch.
Array = NDArray[np.floating]
Tensor = torch.Tensor

class NumpyProcess(Protocol):
    dim: int
    def drift_np(self, x: Array, t: float) -> Array: ...
    def diffusion_np(self, t_scalar: float) -> float | Array: ...
    def transition_mean_std_np(self, x: Array, dt: float, t_scalar: float) -> tuple[Array, Array]: ...

class TorchProcess(Protocol):
    dim: int
    def drift_torch(self, x: Tensor, t: Tensor) -> Tensor: ...
    def diffusion_torch(self, t_scalar: float) -> float | Tensor: ...
    def transition_mean_std_torch(self, x: Tensor, dt: float, t_scalar: float) -> tuple[Tensor, Tensor]: ...
