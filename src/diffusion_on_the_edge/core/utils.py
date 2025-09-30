# diffusion_on_the_edge/core/utils.py
from __future__ import annotations
from typing import Tuple
import numpy as np
import torch

from .types import Array, Tensor


def set_seed(seed: int) -> None:
    """Set both NumPy and Torch PRNG seeds (CPU)."""
    np.random.seed(seed)
    torch.manual_seed(seed)

def ensure_batch_np(x: Array) -> tuple[Array, bool]:
    """Ensure NumPy array is (B, D). Returns (x_batched, was_scalar)."""
    x = np.asarray(x, dtype=float)
    if x.ndim == 1:
        return x[None, :], True
    if x.ndim != 2:
        raise ValueError("Expected x with shape (D,) or (B, D).")
    return x, False

def ensure_batch_torch(x: Tensor) -> tuple[Tensor, bool]:
    """Ensure Torch tensor is (B, D). Returns (x_batched, was_scalar)."""
    if x.dim() == 1:
        return x.unsqueeze(0), True
    if x.dim() != 2:
        raise ValueError("Expected x with shape (D,) or (B, D).")
    return x, False

def time_tensor(value: float, batch: int, *, device: torch.device, dtype: torch.dtype) -> Tensor:
    """Create a (B, 1) torch time tensor filled with `value`."""
    return torch.full((batch, 1), float(value), device=device, dtype=dtype)

def diag_noise_np(
    shape: Tuple[int, int],
    g: float | Array,
    sqrt_dt: float,
) -> Array:
    """Gaussian noise with diagonal std `g * sqrt_dt`.

    Parameters
    ----------
    shape : (B, D)
        Output shape.
    g : float | Array
        Scalar or (D,) vector of per-dim stds at the *current time*.
    sqrt_dt : float
        sqrt(Δt).
    rng : np.random.Generator
        NumPy RNG.
    """
    B, D = shape
    if np.ndim(g) == 0:
        return np.random.normal(size=(B, D)) * (float(g) * sqrt_dt)
    g_vec = np.asarray(g, dtype=float).reshape(1, D)
    return np.random.normal(size=(B, D)) * (g_vec * sqrt_dt)

def diag_noise_torch(
    x: Tensor,
    g_val: float | Tensor,
    sqrt_dt: float,
) -> Tensor:
    """Gaussian noise like `x` with diagonal std `g * sqrt_dt`.

    Parameters
    ----------
    x : (B, D) Tensor
        Reference shape, device, dtype.
    g_val : float | Tensor
        Scalar or (D,) vector of per-dim stds at the *current time*.
    sqrt_dt : float
        sqrt(Δt).
    generator : torch.Generator, optional
        For reproducible sampling.
    """
    if isinstance(g_val, torch.Tensor):
        std = g_val.to(device=x.device, dtype=x.dtype).view(1, -1) * sqrt_dt
        return torch.randn_like(x) * std
    std = float(g_val) * sqrt_dt
    return torch.randn_like(x) * std

def assert_strictly_increasing(ts: Array) -> None:
    """Raise if `ts` is not strictly increasing 1D array."""
    arr = np.asarray(ts, dtype=float).ravel()
    if arr.size < 2 or not np.all(np.diff(arr) > 0.0):
        raise ValueError("Times must be strictly increasing and length >= 2.")
