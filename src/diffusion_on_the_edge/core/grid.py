from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from .types import Array

try:
    import torch
    _TORCH_AVAILABLE = True
except Exception:
    torch = None
    _TORCH_AVAILABLE = False


@dataclass(frozen=True)
class TimeGrid:
    """Closed uniform time grid [t0, t1] with N points.

    Notes
    -----
    - NumPy API: `times() -> (N,)`, `dts() -> (N-1,)`, `step(k) -> (t_k, dt_k)`.
    - Torch API (optional): `times_torch(...)`, `dts_torch(...)`, `time_tensor(...)`.
      These are available only if PyTorch is installed.
    """
    t0: float = 0.0
    t1: float = 1.0
    N: int = 100

    # ---------- NumPy API ----------
    def times(self) -> Array:
        if self.N < 2:
            raise ValueError("TimeGrid.N must be >= 2")
        return np.linspace(self.t0, self.t1, num=self.N, dtype=float)

    def dts(self) -> Array:
        ts = self.times()
        return np.diff(ts)

    def step(self, k: int) -> tuple[float, float]:
        ts = self.times()
        if not (0 <= k < self.N - 1):
            raise IndexError("k must be in [0, N-2]")
        return float(ts[k]), float(ts[k + 1] - ts[k])

    def times_torch(self, device=None, dtype=None):
        """Return times as a torch tensor of shape (N,)."""
        if not _TORCH_AVAILABLE or torch == None:
            raise RuntimeError("PyTorch not available; install torch to use times_torch().")
        
        step = (self.t1 - self.t0) / self.N
        return torch.arange(self.t0, self.t1, step=step, device=device, dtype=dtype)

    def dts_torch(self, device=None, dtype=None):
        """Return forward time steps as a torch tensor of shape (N-1,)."""
        if not _TORCH_AVAILABLE:
            raise RuntimeError("PyTorch not available; install torch to use dts_torch().")
        ts = self.times_torch(device=device, dtype=dtype)
        return ts[1:] - ts[:-1]

    def time_tensor(self, index: int, batch: int, device=None, dtype=None):
        """Return a (batch, 1) torch tensor filled with t[index]."""
        if not _TORCH_AVAILABLE:
            raise RuntimeError("PyTorch not available; install torch to use time_tensor().")
        if not (0 <= index < self.N):
            raise IndexError(f"index must be in [0, {self.N-1}]")
        ts = self.times_torch(device=device, dtype=dtype)
        return ts[index].expand(batch, 1)
