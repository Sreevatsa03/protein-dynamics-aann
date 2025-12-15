from pathlib import Path
from typing import Sequence, Optional

import numpy as np
import matplotlib.pyplot as plt


# consistent colors for model comparison
MODEL_COLORS = {
    "Linear": "C0",
    "Sigmoid": "C1",
    "Mean": "C2",
    "Identity": "C3",
}

MODEL_LINESTYLES = {
    "Linear": "--",
    "Sigmoid": "-",
    "Mean": ":",
    "Identity": "-.",
}


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


def plot_rollout_mse_multi(
    mse_dict: dict[str, np.ndarray],
    save_path: Optional[Path] = None,
):
    """
    Plot rollout MSE versus time for multiple models.

    :param mse_dict: dict mapping model names to MSE arrays over time
    :param save_path: Optional path to save the plot image
    """
    plt.figure(figsize=(8, 5))

    for name, mse_vals in mse_dict.items():
        t = np.arange(len(mse_vals))
        color = MODEL_COLORS.get(name, None)
        ls = MODEL_LINESTYLES.get(name, "-")
        plt.plot(t, mse_vals, label=name, color=color,
                 linestyle=ls, linewidth=1.5)

    plt.xlabel("Time")
    plt.ylabel("MSE")
    plt.legend()
    plt.title("Open-Loop Rollout MSE vs Time")
    plt.tight_layout()

    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=200)

    plt.close()


def plot_per_protein_r2_multi(
    r2_dict: dict[str, np.ndarray],
    save_path: Optional[Path] = None,
    clip_min: Optional[float] = None,
):
    """
    Bar plot comparing per-protein R² for multiple models.

    :param r2_dict: dict mapping model names to R² arrays
    :param save_path: Optional path to save the plot
    :param clip_min: Optional minimum value to clip outliers
    """
    n_models = len(r2_dict)
    d = len(list(r2_dict.values())[0])
    idx = np.arange(d)
    width = 0.8 / n_models

    plt.figure(figsize=(10, 5))

    clipped_info = []
    for i, (name, r2_vals) in enumerate(r2_dict.items()):
        r2_plot = np.copy(r2_vals)

        if clip_min is not None:
            clipped_mask = r2_plot < clip_min
            if np.any(clipped_mask):
                for j in np.where(clipped_mask)[0]:
                    clipped_info.append((j, name, r2_vals[j]))
                r2_plot = np.clip(r2_plot, clip_min, None)

        offset = (i - (n_models - 1) / 2) * width
        color = MODEL_COLORS.get(name, None)
        plt.bar(idx + offset, r2_plot, width, label=name, color=color)

    # mark clipped values
    if len(clipped_info) > 0:
        legend_text = f"Clipped at {clip_min}:\n"
        for protein_idx, model, orig_val in clipped_info:
            legend_text += f"  P{protein_idx} ({model}): {orig_val:.2f}\n"
        plt.text(0.02, 0.02, legend_text.strip(),
                 transform=plt.gca().transAxes, ha="left", va="bottom",
                 fontsize=7, color="red",
                 bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.3))

    plt.xlabel("Protein Index")
    plt.ylabel("R²")
    plt.xticks(idx)
    plt.legend(loc="lower left")
    plt.title("Teacher-Forced Per-Protein R²")
    plt.tight_layout()

    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=200)

    plt.close()


def plot_phase_portrait_comparison(
    true_traj: np.ndarray,
    linear_traj: np.ndarray,
    sigmoid_traj: np.ndarray,
    protein_i: int,
    protein_j: int,
    protein_names: Optional[list[str]] = None,
    save_path: Optional[Path] = None,
):
    """
    Plot phase portrait comparing true vs linear vs sigmoid rollouts.

    :param true_traj: True trajectory (T, state_dim)
    :param linear_traj: Linear model rollout (T, state_dim)
    :param sigmoid_traj: Sigmoid model rollout (T, state_dim)
    :param protein_i: Index of protein for x-axis
    :param protein_j: Index of protein for y-axis
    :param protein_names: Optional list of protein names
    :param save_path: Optional path to save the plot
    """
    plt.figure(figsize=(6, 6))

    plt.plot(true_traj[:, protein_i], true_traj[:, protein_j],
             alpha=0.4, label="True", linewidth=2.0, color="gray")
    plt.plot(linear_traj[:, protein_i], linear_traj[:, protein_j],
             alpha=0.9, label="Linear", linestyle="--", linewidth=1.5)
    plt.plot(sigmoid_traj[:, protein_i], sigmoid_traj[:, protein_j],
             alpha=0.9, label="Sigmoid", linestyle="-", linewidth=1.5)

    # mark starting points
    plt.scatter([true_traj[0, protein_i]], [true_traj[0, protein_j]],
                c='black', s=50, zorder=5, marker='o', label="Start")

    xlabel = protein_names[protein_i] if protein_names else f"Protein {protein_i}"
    ylabel = protein_names[protein_j] if protein_names else f"Protein {protein_j}"

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(f"Phase Portrait: {xlabel} vs {ylabel}")
    plt.legend()
    plt.tight_layout()

    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=200)

    plt.close()


def plot_multi_trajectory_overlay(
    t: np.ndarray,
    true_traj: np.ndarray,
    linear_traj: np.ndarray,
    sigmoid_traj: np.ndarray,
    protein_idx: int,
    protein_name: Optional[str] = None,
    save_path: Optional[Path] = None,
):
    """
    Plot true vs linear vs sigmoid trajectory for a single protein.

    :param t: Array of time points
    :param true_traj: True trajectory (T, state_dim)
    :param linear_traj: Linear model rollout (T, state_dim)
    :param sigmoid_traj: Sigmoid model rollout (T, state_dim)
    :param protein_idx: Index of the protein to plot
    :param protein_name: Optional name of the protein
    :param save_path: Optional path to save the plot
    """
    plt.figure(figsize=(8, 4))

    plt.plot(t, true_traj[:, protein_idx], label="True",
             linewidth=2.0, alpha=0.4, color="gray")
    plt.plot(t, linear_traj[:, protein_idx], label="Linear",
             linestyle="--", alpha=0.9, linewidth=1.5)
    plt.plot(t, sigmoid_traj[:, protein_idx], label="Sigmoid",
             linestyle="-", alpha=0.9, linewidth=1.5)

    plt.xlabel("Time")
    plt.ylabel("Expression")
    title = protein_name if protein_name else f"Protein {protein_idx}"
    plt.title(f"Rollout: {title}")
    plt.legend()
    plt.tight_layout()

    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=200)

    plt.close()


def plot_recovery_distance_vs_iter(
    dist_to_ref_dict: dict[str, np.ndarray],
    dist_to_ref_std_dict: Optional[dict[str, np.ndarray]] = None,
    save_path: Optional[Path] = None,
):
    """
    Plot distance to reference state vs iteration for pattern completion.

    :param dist_to_ref_dict: dict mapping model names to mean distance arrays (K+1,)
    :param dist_to_ref_std_dict: optional dict mapping model names to std arrays
    :param save_path: Optional path to save the plot
    """
    plt.figure(figsize=(8, 5))

    for name, dist in dist_to_ref_dict.items():
        iters = np.arange(len(dist))
        color = MODEL_COLORS.get(name, None)
        ls = MODEL_LINESTYLES.get(name, "-")
        plt.plot(iters, dist, label=name, color=color,
                 linestyle=ls, linewidth=1.5)

        if dist_to_ref_std_dict is not None and name in dist_to_ref_std_dict:
            std = dist_to_ref_std_dict[name]
            plt.fill_between(iters, dist - std, dist +
                             std, alpha=0.15, color=color)

    plt.xlabel("Iteration")
    plt.ylabel("Distance to Reference State")
    plt.title("Pattern Completion: Recovery Distance vs Iteration")
    plt.legend()
    plt.tight_layout()

    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=200)

    plt.close()


def plot_final_recovery_bar(
    final_dist_to_ref: dict[str, float],
    final_dist_to_mean: dict[str, float],
    save_path: Optional[Path] = None,
):
    """
    Bar plot of final recovery distances for each model.

    :param final_dist_to_ref: dict mapping model names to final distance to reference
    :param final_dist_to_mean: dict mapping model names to final distance to global mean
    :param save_path: Optional path to save the plot
    """
    models = list(final_dist_to_ref.keys())
    n_models = len(models)
    idx = np.arange(n_models)
    width = 0.35

    dist_ref = [final_dist_to_ref[m] for m in models]
    dist_mean = [final_dist_to_mean[m] for m in models]

    plt.figure(figsize=(8, 5))
    bars1 = plt.bar(idx - width / 2, dist_ref,
                    width, label="Dist to Reference")
    bars2 = plt.bar(idx + width / 2, dist_mean, width,
                    label="Dist to Global Mean", alpha=0.7)

    # color bars by model
    for i, model in enumerate(models):
        color = MODEL_COLORS.get(model, f"C{i}")
        bars1[i].set_color(color)
        bars2[i].set_color(color)
        bars2[i].set_alpha(0.5)

    plt.xlabel("Model")
    plt.ylabel("Distance (L2 Norm)")
    plt.xticks(idx, models)
    plt.title("Pattern Completion: Final Distances")
    plt.legend()
    plt.tight_layout()

    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=200)

    plt.close()


def plot_recovery_ratio_bar(
    recovery_ratios: dict[str, float],
    save_path: Optional[Path] = None,
):
    """
    Bar plot of recovery ratio (dist_to_mean - dist_to_ref) for each model.
    Positive = model recovers toward reference, Negative = collapses to mean.

    :param recovery_ratios: dict mapping model names to recovery ratio
    :param save_path: Optional path to save the plot
    """
    models = list(recovery_ratios.keys())
    ratios = [recovery_ratios[m] for m in models]
    idx = np.arange(len(models))

    plt.figure(figsize=(7, 5))

    colors = [MODEL_COLORS.get(m, f"C{i}") for i, m in enumerate(models)]
    bars = plt.bar(idx, ratios, color=colors)

    # add horizontal line at zero
    plt.axhline(0, color="black", linewidth=0.8, linestyle="--")

    plt.xlabel("Model")
    plt.ylabel("Recovery Ratio (dist_to_mean - dist_to_ref)")
    plt.xticks(idx, models)
    plt.title("Pattern Completion: Recovery Ratio\n(Positive = recovers to reference, Negative = collapses to mean)")
    plt.tight_layout()

    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=200)

    plt.close()
