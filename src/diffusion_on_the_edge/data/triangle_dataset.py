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
    include_perimeter_normalized_sides: bool = True # (a/l, b/l, c/l) with l = a+b+c


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
    """
    s = np.sort(np.asarray(sides, dtype=float), axis=1)
    a, b, c = s[:, 0], s[:, 1], s[:, 2]
    return np.stack([a / c, b / c], axis=1)


def perimeter_normalized_sides(sides: np.ndarray) -> np.ndarray:
    """
    Normalize sides by perimeter: return (a/l, b/l, c/l) where l = a + b + c.
    """
    s = np.asarray(sides, dtype=float)
    if s.ndim != 2 or s.shape[1] != 3:
        raise ValueError("sides must be a (N,3) array")
    perim = s.sum(axis=1, keepdims=True)
    if np.any(perim <= 0):
        raise ValueError("Perimeter must be positive for all rows.")
    return s / perim


def planar_embedding_coords(sides: np.ndarray) -> np.ndarray:
    """
    Return canonical 2D planar coordinates for the triangle with base c on the x-axis.

    Construction:
      - Place P0 = (0, 0), P1 = (c, 0)
      - Let distances from P2 to P0 and P1 be a and b respectively
      - Then x = (a^2 - b^2 + c^2) / (2c), y = +sqrt(max(a^2 - x^2, 0))
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
    parameters: DatasetOptions,
) -> pd.DataFrame:
    """
    Generate a dataset of triangle side lengths with optional biases and 2D representations.

    The sampling procedure:
      1) Oversample a large valid pool uniformly in (0,1]^3 subject to triangle inequality.
      2) Compute side/angle classes for the pool.
      3) If biases are provided, compute weights and sample n_samples rows.
      4) Assemble the final DataFrame with optional geometric features.

    Returns
    -------
    df : pandas.DataFrame
        Columns always include: 'a', 'b', 'c', 'side_type', 'angle_type'.
    """
    seed = parameters.seed
    n_samples = parameters.n_samples
    oversample_factor = parameters.oversample_factor
    side_bias = parameters.side_bias
    angle_bias = parameters.angle_bias
    sort_sides = parameters.sort_sides
    include_geometry = parameters.include_geometry
    include_normalized_sides = parameters.include_normalized_sides
    include_planar_embedding = parameters.include_planar_embedding
    include_perimeter_normalized_sides = parameters.include_perimeter_normalized_sides

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
    
    if include_perimeter_normalized_sides:
        apl = perimeter_normalized_sides(sides)
        df['a_over_l'] = apl[:, 0]
        df['b_over_l'] = apl[:, 1]
        df['c_over_l'] = apl[:, 2]

    # Deterministic shuffle for the sampling path only
    df = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    return df


# --------------------- Minimal, clean inverse mappers ------------------------

def _mark_validity(sides: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Return per-row masks: finite_mask, pos_mask, tri_mask, valid_mask.
    """
    s = np.asarray(sides, dtype=float)
    finite_mask = np.isfinite(s).all(axis=1)
    pos_mask = (s > 0).all(axis=1)
    a, b, c = s[:, 0], s[:, 1], s[:, 2]
    tri_mask = is_valid_triangle_sides(a, b, c)
    valid_mask = finite_mask & pos_mask & tri_mask
    return finite_mask, pos_mask, tri_mask, valid_mask


def _add_validity_columns(df: pd.DataFrame, sides: np.ndarray) -> pd.DataFrame:
    """
    Adds 'is_valid_triangle' and 'invalid_reason' columns ("" for valid rows).
    """
    finite_mask, pos_mask, tri_mask, valid_mask = _mark_validity(sides)
    df["is_valid_triangle"] = valid_mask

    # Build reasons only for invalid rows
    reasons = np.full(df.shape[0], "", dtype=object)
    bad_idx = np.where(~valid_mask)[0]
    for i in bad_idx:
        if not finite_mask[i]:
            reasons[i] = "nan_or_inf"
        elif not pos_mask[i]:
            reasons[i] = "non_positive"
        elif not tri_mask[i]:
            reasons[i] = "triangle_inequality"
        else:
            reasons[i] = "unknown"
    df["invalid_reason"] = reasons
    return df


def _assemble_dataframe_from_sides(
    sides: np.ndarray,
    seed: Optional[int],
    sort_sides: bool,
    include_geometry: bool,
    include_normalized_sides: bool,
    include_planar_embedding: bool,
    include_perimeter_normalized_sides: bool,
) -> pd.DataFrame:
    """
    Build the final DataFrame from a (N,3) sides array using the same column
    conventions as `generate_triangle_dataset`. No shuffle; preserves order.
    Also adds validity markers.
    """
    s = np.asarray(sides, dtype=float)
    if s.ndim != 2 or s.shape[1] != 3:
        raise ValueError("sides must be a (N,3) array")

    # Validity will be computed on the *reconstructed* (unsorted) sides
    # but we sort for derived features (if requested)
    base_sides = s.copy()

    if sort_sides:
        s = np.sort(s, axis=1)

    # Classes
    st = classify_triangle_sides(s)
    at = classify_triangle_angles(s)

    df = pd.DataFrame(s, columns=["a", "b", "c"])
    df["side_type"] = st
    df["angle_type"] = at

    if include_geometry:
        ang = triangle_angles_degrees(s)
        df["angle_A_deg"] = ang[:, 0]
        df["angle_B_deg"] = ang[:, 1]
        df["angle_C_deg"] = ang[:, 2]
        df["perimeter"] = s.sum(axis=1)
        df["area"] = triangle_area(s)

    if include_normalized_sides:
        norm = normalized_side_coords(s)
        df["a_over_c"] = norm[:, 0]
        df["b_over_c"] = norm[:, 1]

    if include_planar_embedding:
        xy = planar_embedding_coords(s)
        df["x"] = xy[:, 0]
        df["y"] = xy[:, 1]

    if include_perimeter_normalized_sides:
        apl = perimeter_normalized_sides(s)
        df["a_over_l"] = apl[:, 0]
        df["b_over_l"] = apl[:, 1]
        df["c_over_l"] = apl[:, 2]

    # Mark validity using the original (unsorted) reconstructed sides
    df = _add_validity_columns(df, base_sides)
    return df


def dataset_from_perimeter_normalized(
    apl: Union[np.ndarray, list, tuple],
    perimeter: Union[float, np.ndarray, list, tuple],
    parameters: DatasetOptions,
    *,
    renormalize: bool = True,   # set False if your apl rows already sum to 1 exactly
) -> pd.DataFrame:
    """
    Inverse of perimeter normalization: (a/l, b/l, c/l) + l  -> full dataset.
    Preserves input order; does not shuffle. Bad samples are marked, not raised.
    """
    r = np.asarray(apl, dtype=float)
    if r.ndim != 2 or r.shape[1] != 3:
        raise ValueError("apl must be a (N,3) array")
    if renormalize:
        sums = r.sum(axis=1, keepdims=True)
        sums[sums == 0] = 1.0
        r = r / sums

    l = np.asarray(perimeter, dtype=float)
    if np.isscalar(l):
        l = np.full(r.shape[0], float(perimeter), dtype=float)
    if l.ndim != 1 or l.shape[0] != r.shape[0]:
        raise ValueError("perimeter must be a scalar or (N,) matching apl length")
    # Allow non-positive l in marking path? No—scale must be > 0 to be meaningful:
    if (l <= 0).any():
        # Mark later; but we need sides array; set negatives to NaN scale to mark invalid.
        l = np.where(l <= 0, np.nan, l)

    sides = r * l[:, None]

    return _assemble_dataframe_from_sides(
        sides=sides,
        seed=parameters.seed,
        sort_sides=parameters.sort_sides,
        include_geometry=parameters.include_geometry,
        include_normalized_sides=parameters.include_normalized_sides,
        include_planar_embedding=parameters.include_planar_embedding,
        include_perimeter_normalized_sides=parameters.include_perimeter_normalized_sides,
    )


def dataset_from_side_scaled(
    ac_bc: Union[np.ndarray, list, tuple],
    c_values: Union[float, np.ndarray, list, tuple],
    parameters: DatasetOptions,
) -> pd.DataFrame:
    """
    Inverse of longest-side scaling: (a/c, b/c) + c  -> full dataset.
    Preserves input order; does not shuffle. Bad samples are marked, not raised.
    """
    uv = np.asarray(ac_bc, dtype=float)
    if uv.ndim != 2 or uv.shape[1] != 2:
        raise ValueError("ac_bc must be a (N,2) array")

    cs = np.asarray(c_values, dtype=float)
    if np.isscalar(cs):
        cs = np.full(uv.shape[0], float(c_values), dtype=float)
    if cs.ndim != 1 or cs.shape[0] != uv.shape[0]:
        raise ValueError("c_values must be a scalar or (N,) matching ac_bc length")
    if (cs <= 0).any():
        cs = np.where(cs <= 0, np.nan, cs)

    a = uv[:, 0] * cs
    b = uv[:, 1] * cs
    c = cs
    sides = np.stack([a, b, c], axis=1)

    return _assemble_dataframe_from_sides(
        sides=sides,
        seed=parameters.seed,
        sort_sides=parameters.sort_sides,
        include_geometry=parameters.include_geometry,
        include_normalized_sides=parameters.include_normalized_sides,
        include_planar_embedding=parameters.include_planar_embedding,
        include_perimeter_normalized_sides=parameters.include_perimeter_normalized_sides,
    )


def dataset_from_sides(
    sides: Union[np.ndarray, list, tuple],
    parameters: DatasetOptions,
) -> pd.DataFrame:
    """
    Direct build from raw (a,b,c). Preserves input order; does not shuffle.
    Bad samples are marked, not raised.
    """
    s = np.asarray(sides, dtype=float)
    if s.ndim != 2 or s.shape[1] != 3:
        raise ValueError("sides must be a (N,3) array")

    return _assemble_dataframe_from_sides(
        sides=s,
        seed=parameters.seed,
        sort_sides=parameters.sort_sides,
        include_geometry=parameters.include_geometry,
        include_normalized_sides=parameters.include_normalized_sides,
        include_planar_embedding=parameters.include_planar_embedding,
        include_perimeter_normalized_sides=parameters.include_perimeter_normalized_sides,
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
    "perimeter_normalized_sides",
    "generate_triangle_dataset",
    "dataset_from_perimeter_normalized",
    "dataset_from_side_scaled",
    "dataset_from_sides",
]
