import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import animation

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

