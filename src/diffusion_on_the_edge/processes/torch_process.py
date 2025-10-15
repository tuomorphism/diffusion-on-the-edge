from __future__ import annotations
from dataclasses import dataclass
from typing import Union
import torch

from ..core.types import Tensor, TorchProcess

@dataclass(frozen=True)
class OUTorchParams:
    dim: int = 1
    theta: Union[float, Tensor] = 0.5
    mu:    Union[float, Tensor] = 0.0
    sigma: Union[float, Tensor] = 1.0


class OUTorch(TorchProcess):
    """Independent-coordinate OU: dX = θ(μ − X) dt + σ dW (Torch)."""
    def __init__(
        self,
        params: OUTorchParams,
        device: torch.device = torch.device('cpu'),
        dtype: torch.dtype = torch.get_default_dtype(),  # respect global default
    ):
        self.dim = params.dim
        self.theta = torch.as_tensor(params.theta, device=device, dtype=dtype).expand(self.dim).clone()
        self.mu    = torch.as_tensor(params.mu,    device=device, dtype=dtype).expand(self.dim).clone()
        self.sigma = torch.as_tensor(params.sigma, device=device, dtype=dtype).expand(self.dim).clone()

        if torch.any(self.theta <= 0): raise ValueError("theta must be > 0")
        if torch.any(self.sigma <  0): raise ValueError("sigma must be >= 0")

    def drift(self, x: Tensor, _: Tensor) -> Tensor:
        # Cast params to match x (device + dtype), then compute
        theta = self.theta.to(device=x.device, dtype=x.dtype).view(1, -1)
        mu    = self.mu.to(device=x.device,    dtype=x.dtype).view(1, -1)
        return theta * (mu - x)

    def diffusion(self, _: float) -> Tensor:
        # Return sigma matching typical consumer (x) at call sites
        # Consumers (e.g., _g_square / diag_noise_torch) already cast using x;
        # but being explicit prevents surprises if used directly elsewhere.
        return self.sigma  # callers cast with .to(x.device, x.dtype)

    def transition_mean_std(self, x: Tensor, t: Union[float, Tensor]) -> tuple[Tensor, Tensor]:
        # Ensure t, params match x
        device, dtype = x.device, x.dtype
        t      = torch.as_tensor(t, device=device, dtype=dtype)           # shape (), or (B,), or (N,)
        theta  = self.theta.to(device=device, dtype=dtype)                # (D,)
        mu     = self.mu.to(device=device,    dtype=dtype)                # (D,)
        sigma  = self.sigma.to(device=device, dtype=dtype)                # (D,)

        # Broadcast over batch/time as needed: t[..., None] becomes (..., 1) to match D
        e = torch.exp(-t[..., None] * theta)                              # (..., D)
        denom = 2.0 * torch.clamp(theta, min=torch.finfo(dtype).tiny)     # (D,)
        var = (sigma ** 2) * (1.0 - e ** 2) / denom                        # (..., D)
        std = torch.sqrt(torch.clamp(var, min=torch.zeros((), device=device, dtype=dtype)))

        mean = mu + (x - mu) * e                                           # broadcast x with e
        return mean, std
