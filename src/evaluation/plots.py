from pathlib import Path
from typing import Sequence, Optional

import numpy as np
import matplotlib.pyplot as plt


def plot_loss_curves(
    models: dict[str, tuple[Sequence[float], Sequence[float]] | Sequence[float]],
    save_path: Optional[Path] = None,
    plot_train: bool = True,
):
    """
    Plot training and validation loss curves for one or more models.

    :param models: Dictionary mapping model labels to either:
                   - (train_losses, val_losses) tuple for both train and val
                   - val_losses sequence for validation only
                   Example: {"Linear": (train_losses, val_losses), "Sigmoid": val_losses}
    :type models: dict[str, tuple[Sequence[float], Sequence[float]] | Sequence[float]]
    :param save_path: Optional path to save the plot image.
    :type save_path: Path, optional
    :param plot_train: Whether to plot training losses (in addition to validation losses).
    :type plot_train: bool, optional
    """
    plt.figure(figsize=(6, 4))

    for label, losses in models.items():
        # check if losses is a tuple/list with 2 elements or a single sequence
        if isinstance(losses, (tuple, list)) and len(losses) == 2:
            train_losses, val_losses = losses
        else:
            train_losses, val_losses = None, losses

        epochs = np.arange(1, len(val_losses) + 1)
        if plot_train and train_losses is not None:
            plt.plot(epochs, train_losses,
                     label=f"{label} Train", linestyle="--", alpha=0.7)
        plt.plot(epochs, val_losses, label=f"{label} Val")

    plt.xlabel("Epoch")
    plt.ylabel("MSE")
    plt.legend()
    plt.tight_layout()

    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=200)

    plt.close()


def plot_per_protein_metrics(
    metric_linear: np.ndarray,
    metric_sigmoid: np.ndarray,
    metric_name: str,
    save_path: Optional[Path] = None,
    clip_min: Optional[float] = None,
):
    """
    Bar plot comparing per-protein metrics for linear vs sigmoid models.
    :param metric_linear: Any per-protein metricfor the linear model.
    :type metric_linear: np.ndarray
    :param metric_sigmoid: Any per-protein metric for the sigmoid model.
    :type metric_sigmoid: np.ndarray
    :param metric_name: Name of the metric (e.g., "Pearson Correlation").
    :type metric_name: str
    :param save_path: Optional path to save the plot image.
    :type save_path: Path, optional
    :param clip_min: Optional minimum value to clip outliers for better visualization.
    :type clip_min: float, optional
    """
    d = len(metric_linear)
    idx = np.arange(d)
    width = 0.35

    # clip extreme values if specified
    linear_plot = np.copy(metric_linear)
    sigmoid_plot = np.copy(metric_sigmoid)

    clipped_info = []
    if clip_min is not None:
        linear_clipped = linear_plot < clip_min
        sigmoid_clipped = sigmoid_plot < clip_min
        if np.any(linear_clipped) or np.any(sigmoid_clipped):
            for i in np.where(linear_clipped)[0]:
                clipped_info.append((i, 'Linear', metric_linear[i]))
            for i in np.where(sigmoid_clipped)[0]:
                clipped_info.append((i, 'Sigmoid', metric_sigmoid[i]))
            linear_plot = np.clip(linear_plot, clip_min, None)
            sigmoid_plot = np.clip(sigmoid_plot, clip_min, None)

    plt.figure(figsize=(8, 4))
    plt.bar(idx - width / 2, linear_plot, width, label="Linear")
    plt.bar(idx + width / 2, sigmoid_plot, width, label="Sigmoid")

    # mark clipped values with asterisks and show original values
    if len(clipped_info) > 0:
        for protein_idx, model, orig_val in clipped_info:
            plt.text(protein_idx, clip_min, '*', ha='center',
                     va='top', fontsize=16, color='red')

        # add legend for clipped values (above the model legend)
        legend_text = f'Clipped at {clip_min}:\n'
        for protein_idx, model, orig_val in clipped_info:
            legend_text += f'  Protein {protein_idx} ({model}): {orig_val:.1f}\n'
        plt.text(0.02, 0.30, legend_text.strip(),
                 transform=plt.gca().transAxes, ha='left', va='bottom',
                 fontsize=8, color='red', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

    plt.xlabel("Protein Index")
    plt.ylabel(metric_name)
    plt.xticks(idx)

    # Position legend at lower left
    legend = plt.legend(loc='lower left')

    plt.tight_layout()

    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=200)

    plt.close()


def plot_trajectory_overlay(
    t: np.ndarray,
    true_traj: np.ndarray,
    pred_traj: np.ndarray,
    protein_idx: int,
    label: str,
    save_path: Optional[Path] = None,
):
    """
    Plot true vs predicted trajectory for a single protein.

    :param t: Array of time points.
    :type t: np.ndarray
    :param true_traj: True trajectories (time x proteins).
    :type true_traj: np.ndarray
    :param pred_traj: Predicted trajectories (time x proteins).
    :type pred_traj: np.ndarray
    :param protein_idx: Index of the protein to plot.
    :type protein_idx: int
    :param label: Label for the predicted trajectory.
    :type label: str
    :param save_path: Optional path to save the plot image.
    :type save_path: Path, optional
    """
    plt.figure(figsize=(6, 4))
    plt.plot(t, true_traj[:, protein_idx], label="True")
    plt.plot(t, pred_traj[:, protein_idx], label=label)

    # set y-limits based on true trajectory statistics
    true_vals = np.asarray(true_traj[:, protein_idx])
    mean_val = np.mean(true_vals)
    std_val = np.std(true_vals)
    plt.ylim(mean_val - 2 * std_val, mean_val + 2 * std_val)

    plt.xlabel("Time")
    plt.ylabel("Expression")
    plt.title(f"Protein {protein_idx}")
    plt.legend()
    plt.tight_layout()

    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=200)

    plt.close()


def plot_rollout_mse_vs_time(
    mse_linear: np.ndarray,
    mse_sigmoid: np.ndarray,
    save_path: Optional[Path] = None,
):
    """
    Plot rollout MSE versus time for linear and sigmoid models.

    :param mse_linear: MSE values over time for the linear model.
    :type mse_linear: np.ndarray
    :param mse_sigmoid: MSE values over time for the sigmoid model.
    :type mse_sigmoid: np.ndarray
    :param save_path: Optional path to save the plot image.
    :type save_path: Path, optional
    """
    t = np.arange(len(mse_linear))

    plt.figure(figsize=(6, 4))
    plt.plot(t, mse_linear, label="Linear")
    plt.plot(t, mse_sigmoid, label="Sigmoid")
    plt.xlabel("Time")
    plt.ylabel("MSE")
    plt.legend()
    plt.tight_layout()

    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=200)

    plt.close()
