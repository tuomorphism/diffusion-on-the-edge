# triangles2d.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Literal, Optional, Tuple, Union

import numpy as np
import pandas as pd


SideType = Literal["equilateral", "isosceles", "scalene"]
AngleType = Literal["right", "acute", "obtuse"]


@dataclass(frozen=True)
class DatasetOptions:
    """
    Options to control dataset generation.
    """
    n_samples: int = 5_000
    seed: Optional[int] = 42
    sort_sides: bool = True
    oversample_factor: int = 6
    side_bias: Optional[Dict[SideType, float]] = None
    angle_bias: Optional[Dict[AngleType, float]] = None
    include_geometry: bool = True  # area, perimeter, angles
    include_normalized_sides: bool = True  # (a/c, b/c) with c = longest side
    include_planar_embedding: bool = True  # (x, y) with base c on x-axis


def _get_rng(seed: Optional[int]) -> np.random.Generator:
    """Return a NumPy Generator, optionally seeded."""
    return np.random.default_rng(seed)


def is_valid_triangle_sides(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
    """
    Vectorized triangle inequality check.

    Returns
    -------
    mask : np.ndarray of bool
        True where (a, b, c) satisfy triangle inequalities.
    """
    return (a + b > c) & (b + c > a) & (c + a > b)


def sample_valid_triangle_sides(
    n_samples: int,
    seed: Optional[int] = 42,
    sort_sides: bool = True,
    batch_size: Optional[int] = None,
    max_batches: int = 10_000,
) -> np.ndarray:
    """
    Sample valid triangle side lengths (a, b, c) in (0, 1] satisfying the triangle inequality.

    Parameters
    ----------
    n_samples : int
        Number of valid triangles to return.
    seed : int | None
        Random seed for reproducibility.
    sort_sides : bool
        If True, returns sides sorted ascending so that c is the largest side.
    batch_size : int | None
        How many triplets to draw per batch (defaults to 5 * n_samples, capped at 1_000_000).
    max_batches : int
        Safety cap to avoid infinite loops in pathological settings.

    Returns
    -------
    sides : (n_samples, 3) ndarray
        Each row is (a, b, c). If `sort_sides=True`, then a <= b <= c.
    """
    rng = _get_rng(seed)
    bs = batch_size or min(1_000_000, max(1000, 5 * n_samples))
    collected: list[np.ndarray] = []
    batches = 0
    while sum(arr.shape[0] for arr in collected) < n_samples and batches < max_batches:
        raw = rng.random((bs, 3))  # uniform in [0,1)
        if sort_sides:
            raw.sort(axis=1)
        a, b, c = raw[:, 0], raw[:, 1], raw[:, 2]
        mask = is_valid_triangle_sides(a, b, c)
        if mask.any():
            collected.append(raw[mask])
        batches += 1

    if not collected:
        raise RuntimeError("Failed to sample any valid triangles. Try increasing batch_size.")
    out = np.vstack(collected)
    if out.shape[0] < n_samples:
        raise RuntimeError(
            f"Only collected {out.shape[0]} valid triangles; increase batch_size or max_batches."
        )
    return out[:n_samples]


def classify_triangle_sides(
    sides: Union[Tuple[float, float, float], np.ndarray],
    tol: float = 1e-3,
) -> Union[SideType, np.ndarray]:
    """
    Classify by side lengths: 'equilateral', 'isosceles', or 'scalene'.

    Parameters
    ----------
    sides : (3,) or (N,3) array-like
        Triangle side lengths. If not sorted, classification is still correct.
    tol : float
        Absolute tolerance for equality of sides.

    Returns
    -------
    side_type : str or (N,) ndarray[str]
    """
    arr = np.asarray(sides, dtype=float)
    if arr.ndim == 1:
        s = np.sort(arr)
        if np.isclose(s[0], s[1], atol=tol) and np.isclose(s[1], s[2], atol=tol):
            return "equilateral"
        if np.isclose(s[0], s[1], atol=tol) or np.isclose(s[1], s[2], atol=tol):
            return "isosceles"
        return "scalene"
    # vectorized
    s = np.sort(arr, axis=1)
    eq01 = np.isclose(s[:, 0], s[:, 1], atol=tol)
    eq12 = np.isclose(s[:, 1], s[:, 2], atol=tol)
    out = np.full(s.shape[0], "scalene", dtype=object)
    out[eq01 & eq12] = "equilateral"
    out[(eq01 ^ eq12)] = "isosceles"
    return out


def _angles_from_sides_sorted(s: np.ndarray) -> np.ndarray:
    """
    Vectorized angle computation (in degrees) given sorted sides s (a<=b<=c).
    Returns angles (A, B, C) opposite (a, b, c) respectively.
    """
    a, b, c = s[:, 0], s[:, 1], s[:, 2]
    # Law of cosines with clipping for numerical stability
    def safe_acos(x: np.ndarray) -> np.ndarray:
        return np.degrees(np.arccos(np.clip(x, -1.0, 1.0)))

    cosA = (b**2 + c**2 - a**2) / (2 * b * c)
    cosB = (a**2 + c**2 - b**2) / (2 * a * c)
    cosC = (a**2 + b**2 - c**2) / (2 * a * b)

    A = safe_acos(cosA)
    B = safe_acos(cosB)
    C = safe_acos(cosC)
    return np.stack([A, B, C], axis=1)


def triangle_angles_degrees(
    sides: Union[Tuple[float, float, float], np.ndarray]
) -> Union[Tuple[float, float, float], np.ndarray]:
    """
    Compute interior angles (degrees) from side lengths via the law of cosines.

    Parameters
    ----------
    sides : (3,) or (N,3) array-like

    Returns
    -------
    angles_deg : (3,) or (N,3) ndarray
        Angles (A, B, C) opposite (a, b, c) respectively.
    """
    arr = np.asarray(sides, dtype=float)
    if arr.ndim == 1:
        s = np.sort(arr)
        return _angles_from_sides_sorted(s[None, :])[0]
    s = np.sort(arr, axis=1)
    return _angles_from_sides_sorted(s)


def classify_triangle_angles(
    sides: Union[Tuple[float, float, float], np.ndarray],
    tol_deg: float = 1e-1,
) -> Union[AngleType, np.ndarray]:
    """
    Classify by angles: 'right', 'acute', or 'obtuse'.

    Parameters
    ----------
    sides : (3,) or (N,3) array-like
        Triangle side lengths.
    tol_deg : float
        Absolute tolerance in degrees for right-angle detection (default 0.1°).

    Returns
    -------
    angle_type : str or (N,) ndarray[str]
    """
    arr = np.asarray(sides, dtype=float)
    if arr.ndim == 1:
        A, B, C = triangle_angles_degrees(arr)
        max_angle = max(A, B, C)
        if np.isclose(max_angle, 90.0, atol=tol_deg):
            return "right"
        return "acute" if max_angle < 90.0 else "obtuse"
    ang = triangle_angles_degrees(arr)
    max_ang = ang.max(axis=1)
    right = np.isclose(max_ang, 90.0, atol=tol_deg)
    acute = max_ang < 90.0
    out = np.full(arr.shape[0], "obtuse", dtype=object)
    out[acute] = "acute"
    out[right] = "right"
    return out


def normalized_side_coords(sides: np.ndarray) -> np.ndarray:
    """
    Map to 2D 'normalized side-space' by fixing the longest side to 1
    and returning (a/c, b/c) with a<=b<=c.

    Parameters
    ----------
    sides : (N,3) ndarray

    Returns
    -------
    coords : (N,2) ndarray
        Each row is (a/c, b/c).
    """
    s = np.sort(np.asarray(sides, dtype=float), axis=1)
    a, b, c = s[:, 0], s[:, 1], s[:, 2]
    return np.stack([a / c, b / c], axis=1)


def planar_embedding_coords(sides: np.ndarray) -> np.ndarray:
    """
    Return canonical 2D planar coordinates for the triangle with base c on the x-axis.

    Construction:
      - Place P0 = (0, 0), P1 = (c, 0)
      - Let distances from P2 to P0 and P1 be a and b respectively
      - Then x = (a^2 - b^2 + c^2) / (2c), y = +sqrt(max(a^2 - x^2, 0))

    Parameters
    ----------
    sides : (N,3) ndarray
        Side lengths; order free.

    Returns
    -------
    coords : (N,2) ndarray
        Coordinates (x, y) of the third vertex P2; P0 and P1 are fixed as above.
    """
    s = np.sort(np.asarray(sides, dtype=float), axis=1)
    a, b, c = s[:, 0], s[:, 1], s[:, 2]
    x = (a**2 - b**2 + c**2) / (2.0 * c)
    y_sq = np.maximum(a**2 - x**2, 0.0)
    y = np.sqrt(y_sq)
    return np.stack([x, y], axis=1)


def triangle_area(sides: np.ndarray) -> np.ndarray:
    """
    Heron's formula.

    Parameters
    ----------
    sides : (N,3) ndarray

    Returns
    -------
    area : (N,) ndarray
    """
    s = np.asarray(sides, dtype=float)
    p = s.sum(axis=1) / 2.0
    a, b, c = s[:, 0], s[:, 1], s[:, 2]
    area_sq = np.maximum(p * (p - a) * (p - b) * (p - c), 0.0)
    return np.sqrt(area_sq)


def _weights_from_biases(
    side_types: np.ndarray,
    angle_types: np.ndarray,
    side_bias: Optional[Dict[SideType, float]],
    angle_bias: Optional[Dict[AngleType, float]],
) -> np.ndarray:
    """
    Combine independent side/angle biases multiplicatively into a sampling weight.
    Missing keys default to 1.0.
    """
    w = np.ones(side_types.shape[0], dtype=float)
    if side_bias:
        sb = np.vectorize(lambda t: float(side_bias.get(t, 1.0)))(side_types)
        w *= sb
    if angle_bias:
        ab = np.vectorize(lambda t: float(angle_bias.get(t, 1.0)))(angle_types)
        w *= ab
    return w


def generate_triangle_dataset(
    n_samples: int = 5_000,
    side_bias: Optional[Dict[SideType, float]] = None,
    angle_bias: Optional[Dict[AngleType, float]] = None,
    seed: Optional[int] = 42,
    sort_sides: bool = True,
    oversample_factor: int = 6,
    include_geometry: bool = True,
    include_normalized_sides: bool = True,
    include_planar_embedding: bool = True,
) -> pd.DataFrame:
    """
    Generate a dataset of triangle side lengths with optional biases and 2D representations.

    The sampling procedure:
      1) Oversample a large valid pool uniformly in (0,1]^3 subject to triangle inequality.
      2) Compute side/angle classes for the pool.
      3) If biases are provided, compute weights:
            weight = side_bias[side_type] * angle_bias[angle_type]
         and sample `n_samples` rows from the pool with probability ∝ weight.
         If no biases, sample uniformly from the pool.
      4) Assemble the final DataFrame with optional geometric features.

    Parameters
    ----------
    n_samples : int
        Number of samples to draw for the final dataset.
    side_bias : dict | None
        e.g. {'equilateral': 0.02, 'isosceles': 0.18, 'scalene': 0.80}
    angle_bias : dict | None
        e.g. {'right': 0.05, 'acute': 0.70, 'obtuse': 0.25}
    seed : int | None
        Seed for reproducibility.
    sort_sides : bool
        If True, stored sides are sorted ascending (a <= b <= c).
    oversample_factor : int
        Pool size multiplier; pool_size = oversample_factor * n_samples.
        Increase if your biases are extreme and underrepresented in a modest pool.
    include_geometry : bool
        If True, include perimeter, area, and individual angles (degrees).
    include_normalized_sides : bool
        If True, include columns 'a_over_c', 'b_over_c'.
    include_planar_embedding : bool
        If True, include columns 'x', 'y' for the canonical planar embedding.

    Returns
    -------
    df : pandas.DataFrame
        Columns always include: 'a', 'b', 'c', 'side_type', 'angle_type'.
        Optional columns per flags above.
    """
    rng = _get_rng(seed)

    pool_size = max(n_samples * oversample_factor, n_samples + 1000)
    pool = sample_valid_triangle_sides(
        n_samples=pool_size, seed=seed, sort_sides=True  # sorted for consistent derived features
    )

    # Side/angle classes
    side_types = classify_triangle_sides(pool)
    angle_types = classify_triangle_angles(pool)

    # Compute sampling weights if biases specified
    w = _weights_from_biases(side_types, angle_types, side_bias, angle_bias)
    if np.allclose(w, 0.0):
        raise ValueError("All bias weights are zero; provide at least one positive weight.")
    if not np.all(w >= 0):
        raise ValueError("Bias weights must be non-negative.")

    # Sample final indices
    probs = w / w.sum()
    idx = rng.choice(pool.shape[0], size=n_samples, replace=False if n_samples <= pool.shape[0] else True, p=probs)
    sides = pool[idx]
    st = side_types[idx]
    at = angle_types[idx]

    # Optionally sort sides for output
    if sort_sides:
        sides.sort(axis=1)

    df = pd.DataFrame(sides, columns=["a", "b", "c"])
    df["side_type"] = st
    df["angle_type"] = at

    if include_geometry:
        ang = triangle_angles_degrees(sides)
        df["angle_A_deg"] = ang[:, 0]
        df["angle_B_deg"] = ang[:, 1]
        df["angle_C_deg"] = ang[:, 2]
        df["perimeter"] = sides.sum(axis=1)
        df["area"] = triangle_area(sides)

    if include_normalized_sides:
        norm = normalized_side_coords(sides)
        df["a_over_c"] = norm[:, 0]
        df["b_over_c"] = norm[:, 1]

    if include_planar_embedding:
        xy = planar_embedding_coords(sides)
        df["x"] = xy[:, 0]
        df["y"] = xy[:, 1]

    # Shuffle for good measure (deterministic via seed)
    df = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    return df


# Convenience aliases mirroring your original API
def generate_biased_triangle_dataset(
    n_samples: int = 5_000,
    side_bias: Optional[Dict[SideType, float]] = None,
    angle_bias: Optional[Dict[AngleType, float]] = None,
    seed: Optional[int] = 42,
) -> pd.DataFrame:
    """
    Backwards-compatible wrapper that calls `generate_triangle_dataset` with geometry and 2D features enabled.
    """
    return generate_triangle_dataset(
        n_samples=n_samples,
        side_bias=side_bias,
        angle_bias=angle_bias,
        seed=seed,
        sort_sides=True,
        oversample_factor=6,
        include_geometry=True,
        include_normalized_sides=True,
        include_planar_embedding=True,
    )


__all__ = [
    "DatasetOptions",
    "is_valid_triangle_sides",
    "sample_valid_triangle_sides",
    "classify_triangle_sides",
    "classify_triangle_angles",
    "triangle_angles_degrees",
    "triangle_area",
    "normalized_side_coords",
    "planar_embedding_coords",
    "generate_triangle_dataset",
    "generate_biased_triangle_dataset",
]
