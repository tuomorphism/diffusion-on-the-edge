from __future__ import annotations
from dataclasses import dataclass
from typing import Callable
import torch

from ..core.types import Tensor, TorchProcess

@dataclass(frozen=True)
class OUTorchParams:
    dim: int = 1
    theta: float | Tensor = 0.5
    mu: float | Tensor = 0.0
    sigma: float | Tensor = 1.0


class OUTorch(TorchProcess):
    """Independent-coordinate OU: dX = θ(μ − X) dt + σ dW (Torch)."""
    def __init__(self, params: OUTorchParams, device: torch.device = torch.device('cpu'), dtype: torch.dtype = torch.float):
        self.dim = params.dim
        self.theta = torch.as_tensor(params.theta, device=device, dtype=dtype).expand(self.dim)
        self.mu = torch.as_tensor(params.mu, device=device, dtype=dtype).expand(self.dim)
        self.sigma = torch.as_tensor(params.sigma, device=device, dtype=dtype).expand(self.dim)
        if torch.any(self.theta <= 0): raise ValueError("theta must be > 0")
        if torch.any(self.sigma < 0):  raise ValueError("sigma must be >= 0")

    def drift(self, x: Tensor, _: Tensor) -> Tensor:
        return self.theta.view(1, -1) * (self.mu.view(1, -1) - x)

    def diffusion(self, _: float) -> Tensor:
        return self.sigma 

    def transition_mean_std(self, x: Tensor, t: Tensor) -> tuple[Tensor, Tensor]:
        e = torch.exp(-t[..., None] * self.theta)  # broadcasts theta over last dim
        denom = 2.0 * torch.clamp(self.theta, min=1e-12)
        var   = (self.sigma**2) * (1.0 - e**2) / denom
        std   = torch.sqrt(torch.clamp(var, min=0.0))
        mean = self.mu + (x - self.mu) * e
        return mean, std
 
