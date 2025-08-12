import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

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
        sns.lineplot(x=time, y=data[:, i], label=lbl, linewidth=1.5, ax = ax)

    plt.xlabel("Time")
    plt.ylabel("x(t)")
    plt.title(title)
    plt.tight_layout()
    return (fig, ax)
 