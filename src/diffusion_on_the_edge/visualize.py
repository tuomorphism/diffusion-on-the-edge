from typing import Callable, Dict, Tuple, Optional

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import animation


# Defining some constants for triangle transformations
Va = np.array([0.0, 0.0])
Vb = np.array([1.0, 0.0])
Vc = np.array([0.5, np.sqrt(3)/2])
T = np.array([Vb - Va, Vc - Va]).T  # 2x2
invT = np.linalg.inv(T)


def plot_multiple_1d_trajectories(data, time=None, labels=None, title='Multiple 1D Diffusion Trajectories'):
    """
    Plot multiple 1D diffusion trajectories.

    Parameters:
    - data: numpy array of shape (T, N), where each column is a separate trajectory
    - time: optional array of shape (T,) for time values. If None, will use np.arange(T)
    - labels: optional list of N labels for the trajectories
    - title: plot title
    """
    T, N = data.shape
    if time is None:
        time = np.arange(T)

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.set_theme(style="whitegrid")

    for i in range(N):
        lbl = labels[i] if labels is not None else f"Trajectory {i}"
        sns.lineplot(x=time, y=data[:, i], label=lbl, linewidth=1.5, ax = ax).set(xlim=(time[0], time[-1]))

    plt.xlabel("Time")
    plt.ylabel("x(t)")
    plt.title(title)
    plt.tight_layout()
    return (fig, ax)

def plot_score_field(
    score,
    domain=None,
    grid_points=25,
    title="Score Field",
    ax=None,
):
    """
    Visualize a score function field.

    Parameters
    ----------
    score : callable
        Function mapping x -> score(x).
        - 1D: x has shape (N, 1) or (N,), score returns (N, 1) or (N,)
        - 2D: x has shape (N, 2), score returns (N, 2)
    domain : tuple or None
        - 1D: (xmin, xmax)
        - 2D: ((xmin, xmax), (ymin, ymax))
        If None, defaults to (-3, 3) for 1D or ((-3, 3), (-3, 3)) for 2D (auto-detected).
    grid_points : int
        Number of grid points per axis.
    title : str
        Plot title.
    ax : matplotlib.axes.Axes or None
        If provided, draw on this axis; otherwise create a new figure/axes.

    Returns
    -------
    (fig, ax)
    """
    sns.set_theme(style="whitegrid")

    # --- Infer dimensionality by probing score ---
    # Try a 2D probe first; fall back to 1D if it fails.
    dim = None
    try:
        test = np.array([[0.0, 0.0]])
        out = np.asarray(score(test))
        if out.ndim == 2 and out.shape[1] == 2:
            dim = 2
    except Exception as e:
        print(f'Exception {e}!')
    if dim is None:
        try:
            test = np.array([[0.0]])
            out = np.asarray(score(test))
            if out.ndim == 2 and out.shape[1] in (1,):
                dim = 1
            elif out.ndim == 1:
                dim = 1
        except Exception as e:
            print(f"Exception {e}")

    if dim is None:
        raise ValueError("Could not infer input/output dimensionality for `score` (expected 1D or 2D).")

    # --- Create figure/axes ---
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    else:
        fig = ax.figure

    if dim == 1:
        if domain is None:
            domain = (-3.0, 3.0)
        xmin, xmax = domain
        xs = np.linspace(xmin, xmax, grid_points)
        X = xs.reshape(-1, 1)

        S = np.asarray(score(X)).reshape(-1)
        sns.lineplot(x=xs, y=S, linewidth=2.0, ax=ax)
        ax.axhline(0.0, color="black", linewidth=1, alpha=0.6)
        ax.set_xlabel("x")
        ax.set_ylabel("score(x)")
        ax.set_title(title)
        ax.set_xlim(xmin, xmax)

    else:
        if domain is None:
            domain = ((-3.0, 3.0), (-3.0, 3.0))
        (xmin, xmax), (ymin, ymax) = domain

        xs = np.linspace(xmin, xmax, grid_points)
        ys = np.linspace(ymin, ymax, grid_points)
        XX, YY = np.meshgrid(xs, ys)
        pts = np.stack([XX.ravel(), YY.ravel()], axis=1)

        S = np.asarray(score(pts))
        if S.ndim != 2 or S.shape[1] != 2:
            raise ValueError("For 2D visualization, score must return shape (N, 2).")

        U = S[:, 0].reshape(XX.shape)
        V = S[:, 1].reshape(XX.shape)
        M = np.sqrt(U**2 + V**2)

        # Background magnitude for context
        # (use pcolormesh for speed & a soft alpha to keep grid visible)
        pcm = ax.pcolormesh(XX, YY, M, shading="auto", alpha=0.35, cmap="viridis")

        # Normalize arrows for readability, keep relative direction
        eps = 1e-8
        Un = U / (M + eps)
        Vn = V / (M + eps)

        # Scale arrows based on domain size
        span = max((xmax - xmin), (ymax - ymin))
        scale = grid_points / (0.35 * span)  # heuristic for decent arrow density/length

        _ = ax.quiver(XX, YY, Un, Vn, angles="xy", scale_units="xy", scale=scale, width=0.003, headwidth=3)

        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_title(title)
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        plt.colorbar(pcm, ax=ax, label="||score(x)||")

    plt.tight_layout()
    return (fig, ax)


def animate_score_field(
    score,
    domain=None,
    grid_points=25,
    frames=120,
    interval=40,
    title="Score Field Animation",
    time_values=None,
    ax=None,
):
    """
    Create an animation of a score function field.

    Parameters
    ----------
    score : callable
        Score function. Two accepted signatures:
          - time-independent: score(X) -> array of shape (N, d)
          - time-dependent:   score(X, t) -> array of shape (N, d)
        X is an array of shape (N, d): d in {1,2}.
    domain : tuple or None
        - 1D: (xmin, xmax)
        - 2D: ((xmin, xmax), (ymin, ymax))
        Defaults to (-3,3) or ((-3,3),(-3,3)) if None.
    grid_points : int
        Grid resolution per axis.
    frames : int
        Number of animation frames.
    interval : int
        Delay between frames in milliseconds.
    title : str
        Figure title.
    time_values : array-like or None
        Sequence of time values passed to score(X, t). If None, uses np.linspace(0, 1, frames).
        Ignored if the score is time-independent.
    ax : matplotlib.axes.Axes or None
        If provided, draw on this axis.

    Returns
    -------
    (fig, anim)
        fig  : matplotlib Figure
        anim : matplotlib.animation.FuncAnimation
    """
    sns.set_theme(style="whitegrid")

    # --- Detect dimensionality & time-dependence ---
    dim = None
    time_dependent = False

    # Try 2D first
    try:
        testX2 = np.array([[0.0, 0.0]])
        s2 = np.asarray(score(testX2))
        if s2.ndim == 2 and s2.shape[1] == 2:
            dim = 2
    except Exception:
        pass

    if dim is None:
        # Try 1D
        try:
            testX1 = np.array([[0.0]])
            s1 = np.asarray(score(testX1))
            if s1.ndim == 2 and s1.shape[1] == 1:
                dim = 1
            elif s1.ndim == 1:
                dim = 1
        except Exception as e:
            print(f"Exception {e}")

    if dim is None:
        # Try time-dependent signatures
        try:
            testX2 = np.array([[0.0, 0.0]])
            s2t = np.asarray(score(testX2, 0.0))
            if s2t.ndim == 2 and s2t.shape[1] == 2:
                dim = 2
                time_dependent = True
        except Exception as e:
            print(f"Exception {e}")

    if dim is None:
        try:
            testX1 = np.array([[0.0]])
            s1t = np.asarray(score(testX1, 0.0))
            if (s1t.ndim == 2 and s1t.shape[1] == 1) or (s1t.ndim == 1):
                dim = 1
                time_dependent = True
        except Exception as e:
            print(f"Exception {e}")

    if dim is None:
        raise ValueError("Could not infer 1D/2D or time dependence from `score`.")

    if time_values is None:
        time_values = np.linspace(0.0, 1.0, frames)

    # --- Prepare figure/axes ---
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    else:
        fig = ax.figure

    # --- 1D setup ---
    if dim == 1:
        if domain is None:
            domain = (-3.0, 3.0)
        xmin, xmax = domain
        xs = np.linspace(xmin, xmax, grid_points)
        X = xs.reshape(-1, 1)

        # initial field
        if time_dependent:
            S = np.asarray(score(X, time_values[0])).reshape(-1)
        else:
            S = np.asarray(score(X)).reshape(-1)

        line, = ax.plot(xs, S, linewidth=2.0)
        ax.axhline(0.0, color="black", linewidth=1, alpha=0.6)
        ax.set_xlim(xmin, xmax)
        ax.set_xlabel("x")
        ax.set_ylabel("score(x, t)" if time_dependent else "score(x)")
        ax.set_title(title)

        def init():
            line.set_ydata(S)
            return (line,)

        def update(i):
            if time_dependent:
                Si = np.asarray(score(X, time_values[i])).reshape(-1)
            else:
                Si = S
            line.set_ydata(Si)
            return (line,)

        anim = animation.FuncAnimation(
            fig, update, init_func=init, frames=frames, interval=interval, blit=True
        )

    # --- 2D setup ---
    else:
        if domain is None:
            domain = ((-3.0, 3.0), (-3.0, 3.0))
        (xmin, xmax), (ymin, ymax) = domain

        xs = np.linspace(xmin, xmax, grid_points)
        ys = np.linspace(ymin, ymax, grid_points)
        XX, YY = np.meshgrid(xs, ys)
        P = np.stack([XX.ravel(), YY.ravel()], axis=1)

        # initial field
        if time_dependent:
            S = np.asarray(score(P, time_values[0]))
        else:
            S = np.asarray(score(P))

        if S.ndim != 2 or S.shape[1] != 2:
            raise ValueError("For 2D visualization, score must return shape (N, 2).")

        U = S[:, 0].reshape(XX.shape)
        V = S[:, 1].reshape(XX.shape)
        M = np.sqrt(U**2 + V**2)

        # background magnitude
        pcm = ax.pcolormesh(XX, YY, M, shading="auto", alpha=0.35, cmap="viridis")

        # normalized arrows for readability
        eps = 1e-8
        Un = U / (M + eps)
        Vn = V / (M + eps)

        span = max((xmax - xmin), (ymax - ymin))
        scale = grid_points / (0.35 * span)

        q = ax.quiver(XX, YY, Un, Vn, angles="xy", scale_units="xy", scale=scale, width=0.003, headwidth=3)
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_title(title)
        plt.colorbar(pcm, ax=ax, label="||score(x, t)||" if time_dependent else "||score(x)||")

        def init():
            pcm.set_array(M.ravel())
            q.set_UVC(Un, Vn)
            return (pcm, q)

        def update(i):
            if time_dependent:
                Si = np.asarray(score(P, time_values[i]))
            else:
                Si = S
            Ui = Si[:, 0].reshape(XX.shape)
            Vi = Si[:, 1].reshape(XX.shape)
            Mi = np.sqrt(Ui**2 + Vi**2)
            pcm.set_array(Mi.ravel())

            Ui_n = Ui / (Mi + eps)
            Vi_n = Vi / (Mi + eps)
            q.set_UVC(Ui_n, Vi_n)
            return (pcm, q)

        anim = animation.FuncAnimation(
            fig, update, init_func=init, frames=frames, interval=interval, blit=False
        )

    plt.tight_layout()
    return (fig, anim)



def normalize_perimeter(abc: np.ndarray) -> np.ndarray:
    """Return perimeter-normalized sides xyz = abc / (a+b+c).
    abc: (N,3) > 0
    """
    s = abc.sum(axis=1, keepdims=True)
    return abc / s


def bary_to_xy(xyz: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Map barycentric (x,y,z) on the unit simplex to Cartesian (X,Y).
    xyz: (N,3) with x+y+z=1 and x,y,z >= 0
    Returns: (X, Y) each shape (N,)
    """
    x, y, z = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    X = x * Va[0] + y * Vb[0] + z * Vc[0]
    Y = x * Va[1] + y * Vb[1] + z * Vc[1]
    return X, Y


def xy_to_bary(X: np.ndarray, Y: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Map Cartesian (X,Y) inside the equilateral triangle to barycentric (x,y,z).
    Returns three arrays x,y,z with the same shape as X,Y. Points outside the
    triangle will give some negatives; mask them with (x>0)&(y>0)&(z>0).
    """
    P = np.stack([X - Va[0], Y - Va[1]], axis=-1)  # (...,2)
    yz = P @ invT.T
    y, z = yz[..., 0], yz[..., 1]
    x = 1.0 - y - z
    return x, y, z


# ---- Masks and grid ----

def masks_from_bary(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute inside-simplex, valid, and non-valid masks from barycentric coords.
    Valid triangles (triangle inequality) in normalized coords: x,y,z < 0.5.
    """
    inside = (x > 0) & (y > 0) & (z > 0)
    valid = inside & (x < 0.5) & (y < 0.5) & (z < 0.5)
    nonvalid = inside & ~valid
    return inside, valid, nonvalid


def build_simplex_grid(nx: int = 420, ny: int = 364):
    """Create a regular XY grid covering the equilateral triangle bounding box.
    Returns dict with X,Y,xg,yg,zg,inside,valid,nonvalid and axis extents.
    """
    Xg = np.linspace(0.0, 1.0, nx)
    Yg = np.linspace(0.0, np.sqrt(3)/2, ny)
    X, Y = np.meshgrid(Xg, Yg)
    xg, yg, zg = xy_to_bary(X, Y)
    inside, valid, nonvalid = masks_from_bary(xg, yg, zg)
    return {
        "X": X,
        "Y": Y,
        "xg": xg,
        "yg": yg,
        "zg": zg,
        "inside": inside,
        "valid": valid,
        "nonvalid": nonvalid,
        "extent": (0.0, 1.0, 0.0, float(np.sqrt(3)/2)),
        "X_axis": Xg,
        "Y_axis": Yg,
    }


# ---- Density via scale marginalization ----
LogPFn = Callable[[np.ndarray], np.ndarray]

def log_p_xy(
    model_logp_abc: LogPFn,
    x_flat: np.ndarray,
    y_flat: np.ndarray,
    s_samples: np.ndarray,
    mask_outside_simplex: bool = True,
) -> np.ndarray:
    """Monte Carlo estimate of log p(x,y) on the simplex after marginalizing scale s.
    p_{x,y}(x,y) = ∫ p_{a,b,c}(x s, y s, z s) s^2 ds, z=1-x-y.

    Returns array of shape like x_flat with -inf outside simplex if masked.
    """
    z_flat = 1.0 - x_flat - y_flat
    inside = (x_flat > 0) & (y_flat > 0) & (z_flat > 0)

    L = []
    for s in np.atleast_1d(s_samples):
        abc = np.column_stack([x_flat * s, y_flat * s, z_flat * s])
        lp = model_logp_abc(abc) + 2.0 * np.log(s)  # Jacobian s^2
        L.append(lp)
    L = np.stack(L, axis=1)  # (M, K)
    m = np.max(L, axis=1, keepdims=True)
    logmean = m.squeeze() + np.log(np.mean(np.exp(L - m), axis=1))

    if mask_outside_simplex:
        logmean = logmean.astype(float)
        logmean[~inside] = -np.inf
    return logmean


def evaluate_models_on_grid(
    base_logp_abc: LogPFn,
    fine_logp_abc: LogPFn,
    s_samples: np.ndarray,
    grid: Optional[Dict[str, np.ndarray]] = None,
) -> Dict[str, np.ndarray]:
    """Evaluate both models' log density over the simplex grid.
    Returns a dict with logd_base, logd_fine, dlog (fine−base), vmin, vmax, and masks.
    """
    if grid is None:
        grid = build_simplex_grid()
    X, Y, xg, yg = grid["X"], grid["Y"], grid["xg"], grid["yg"]

    logd_base = log_p_xy(base_logp_abc, xg.ravel(), yg.ravel(), s_samples).reshape(X.shape)
    logd_fine = log_p_xy(fine_logp_abc, xg.ravel(), yg.ravel(), s_samples).reshape(X.shape)

    inside = grid["inside"]
    finite = np.isfinite(logd_base) & np.isfinite(logd_fine) & inside
    vmin = float(np.min([logd_base[finite].min(), logd_fine[finite].min()]))
    vmax = float(np.max([logd_base[finite].max(), logd_fine[finite].max()]))

    return {
        **grid,
        "logd_base": logd_base,
        "logd_fine": logd_fine,
        "dlog": logd_fine - logd_base,
        "vmin": vmin,
        "vmax": vmax,
    }


# ---- Plotting ----

def plot_density_panels(
    evals: Dict[str, np.ndarray],
    titles: Tuple[str, str, str] = (
        "Base: log density",
        "Fine: log density",
        "Δ log density (fine − base)",
    ),
    cmap_main: str = "viridis",
    cmap_diff: str = "RdBu_r",
):
    """Three-panel plot: base, fine, and difference with a validity contour."""
    X, Y = evals["X"], evals["Y"]
    extent = evals["extent"]
    valid = evals["valid"]
    logd_base, logd_fine, dlog = evals["logd_base"], evals["logd_fine"], evals["dlog"]
    vmin, vmax = evals["vmin"], evals["vmax"]

    fig, axes = plt.subplots(1, 3, figsize=(14, 4), sharex=True, sharey=True)
    im0 = axes[0].imshow(logd_base, origin="lower", extent=extent, vmin=vmin, vmax=vmax, aspect="equal", cmap=cmap_main)
    axes[0].set_title(titles[0])
    axes[0].contour(X, Y, valid, levels=[0.5], colors="k", linewidths=1.1)

    im1 = axes[1].imshow(logd_fine, origin="lower", extent=extent, vmin=vmin, vmax=vmax, aspect="equal", cmap=cmap_main)
    axes[1].set_title(titles[1])
    axes[1].contour(X, Y, valid, levels=[0.5], colors="k", linewidths=1.1)

    im2 = axes[2].imshow(dlog, origin="lower", extent=extent, aspect="equal", cmap=cmap_diff)
    axes[2].set_title(titles[2])
    axes[2].contour(X, Y, valid, levels=[0.5], colors="k", linewidths=1.1)

    cbar_main = fig.colorbar(im1, ax=axes[:2], fraction=0.046, pad=0.04)
    cbar_main.set_label("log density")
    fig.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04).set_label("Δ")

    for ax in axes:
        ax.set_xlim(extent[0], extent[1])
        ax.set_ylim(extent[2], extent[3])
        ax.set_aspect("equal", adjustable="box")
    plt.tight_layout()
    return fig, axes


# ---- Mass fraction utilities ----

def _stable_area_sum(logd: np.ndarray, mask: np.ndarray, cell_area: float) -> float:
    vals = logd.copy()
    vals[~mask] = -np.inf
    # subtract max for stability
    m = np.nanmax(vals[mask]) if np.any(mask) else -np.inf
    if not np.isfinite(m):
        return 0.0
    return float(np.sum(np.exp(vals[mask] - m)) * np.exp(m) * cell_area)


def invalid_mass_fraction(evals: Dict[str, np.ndarray], which: str = "base") -> float:
    """Estimate fraction of probability mass in the non-valid region.
    which ∈ {"base", "fine"}
    """
    logd = evals["logd_base"] if which == "base" else evals["logd_fine"]
    inside = evals["inside"]
    nonvalid = evals["nonvalid"]
    X_axis, Y_axis = evals["X_axis"], evals["Y_axis"]
    dx = float(X_axis[1] - X_axis[0])
    dy = float(Y_axis[1] - Y_axis[0])
    cell_area = dx * dy
    num = _stable_area_sum(logd, nonvalid, cell_area)
    den = _stable_area_sum(logd, inside, cell_area)
    return 0.0 if den == 0.0 else num / den


def hexbin_counts_from_samples(abc_samps: np.ndarray, gridsize: int = 90, ax = None):
    """Hexbin counts of perimeter-normalized samples on the simplex (valid+non-valid)."""
    xyz = normalize_perimeter(abc_samps)
    X, Y = bary_to_xy(xyz)
    if ax is None:
        fig, ax = plt.subplots(figsize=(5, 4))
    hb = ax.hexbin(X, Y, gridsize=gridsize)
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, np.sqrt(3)/2)
    ax.set_aspect("equal", adjustable="box")
    ax.set_title("Heatmap (perimeter-normalized samples)")
    plt.colorbar(hb, ax=ax, label="count")
    return ax


def scatter_valid_vs_nonvalid_samples(abc_samps: np.ndarray, title: str = "Samples: valid vs non-valid", s: int = 6, alpha: float = 0.55):
    """Scatter plot of samples colored by validity in normalized space."""
    xyz = normalize_perimeter(abc_samps)
    x, y, z = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    inside, valid, nonvalid = masks_from_bary(x, y, z)
    X, Y = bary_to_xy(xyz)

    fig, ax = plt.subplots(figsize=(5, 4))
    ax.scatter(X[nonvalid], Y[nonvalid], s=s, alpha=alpha, label="non-valid")
    ax.scatter(X[valid], Y[valid], s=s, alpha=alpha, label="valid")
    ax.legend()
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, np.sqrt(3)/2)
    ax.set_aspect("equal", adjustable="box")
    ax.set_title(title)
    return ax


def delta_logp_hexbin(abc_eval: np.ndarray, delta_logp: np.ndarray, gridsize: int = 90, title: str = "Mean Δ log p on simplex"):
    """Hexbin of per-sample Δlog p over the simplex (use reduce_C_function=np.mean)."""
    assert abc_eval.shape[0] == delta_logp.shape[0], "Mismatched N"
    xyz = normalize_perimeter(abc_eval)
    X, Y = bary_to_xy(xyz)
    fig, ax = plt.subplots(figsize=(5, 4))
    hb = ax.hexbin(X, Y, C=delta_logp, gridsize=gridsize, reduce_C_function=np.mean, cmap="viridis")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, np.sqrt(3)/2)
    ax.set_aspect("equal", adjustable="box")
    ax.set_title(title)
    plt.colorbar(hb, ax=ax, label="mean Δ log p")
    return ax
