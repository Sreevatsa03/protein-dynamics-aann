"""
Dynamical diagnostics for evaluating simulated protein trajectories.

This module provides a single function `evaluate_dynamics` that computes
a fixed set of diagnostics to help select generator regimes that produce
rich, protein-like dynamics suitable for AANN experiments.
"""

from __future__ import annotations

import numpy as np
from scipy.signal import hilbert
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import pdist
from sklearn.decomposition import PCA
import torch


def _to_numpy(x: np.ndarray | torch.Tensor) -> np.ndarray:
    """Convert tensor to numpy array if needed."""
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def _spectral_diagnostics(X: np.ndarray, dt: float) -> dict:
    """
    Compute spectral coherence metrics.

    :param X: trajectories (n_rollouts, T, N)
    :param dt: time step
    :return: dict with spectral metrics
    """
    n_rollouts, T, N = X.shape

    # FFT per rollout, per node, then average power spectrum
    freqs = np.fft.rfftfreq(T, d=dt)
    power_avg = np.zeros((N, len(freqs)))

    for r in range(n_rollouts):
        for n in range(N):
            sig = X[r, :, n] - X[r, :, n].mean()
            fft_vals = np.fft.rfft(sig)
            power_avg[n] += np.abs(fft_vals) ** 2

    power_avg /= n_rollouts

    # dominant frequency per node (skip DC)
    dom_freq = np.zeros(N)
    sharpness = np.zeros(N)

    for n in range(N):
        power_n = power_avg[n, 1:]  # skip DC
        freqs_n = freqs[1:]
        total_power = power_n.sum() + 1e-12
        peak_idx = np.argmax(power_n)
        dom_freq[n] = freqs_n[peak_idx]
        sharpness[n] = power_n[peak_idx] / total_power

    return {
        "spectral_mean_freq": float(np.mean(dom_freq)),
        "spectral_freq_std": float(np.std(dom_freq)),
        "spectral_sharpness": float(np.mean(sharpness)),
    }


def _saturation_diagnostics(X: np.ndarray, eps: float = 0.05) -> dict:
    """
    Compute fraction of time spent near bounds [0, eps] ∪ [1-eps, 1].

    :param X: trajectories (n_rollouts, T, N)
    :param eps: boundary threshold
    :return: dict with saturation metrics
    """
    near_bounds = (X < eps) | (X > (1 - eps))
    # fraction per node, averaged over rollouts and time
    sat_per_node = near_bounds.mean(axis=(0, 1))  # (N,)

    return {
        "mean_saturation": float(np.mean(sat_per_node)),
        "saturation_std": float(np.std(sat_per_node)),
    }


def _phase_diagnostics(X: np.ndarray) -> dict:
    """
    Compute phase-lag matrix via Hilbert transform.

    :param X: trajectories (n_rollouts, T, N)
    :return: dict with phase metrics and phase-lag matrix
    """
    n_rollouts, T, N = X.shape

    phase_lag_accum = np.zeros((N, N))

    for r in range(n_rollouts):
        phases = np.zeros((T, N))
        for n in range(N):
            sig = X[r, :, n] - X[r, :, n].mean()
            analytic = hilbert(sig)
            phases[:, n] = np.angle(analytic)

        # pairwise phase differences, averaged over time
        for i in range(N):
            for j in range(N):
                # circular mean of phase difference
                diff = phases[:, i] - phases[:, j]
                mean_diff = np.arctan2(
                    np.sin(diff).mean(), np.cos(diff).mean())
                phase_lag_accum[i, j] += mean_diff

    phase_lag_matrix = phase_lag_accum / n_rollouts

    # phase order score: mean absolute off-diagonal lag
    mask = ~np.eye(N, dtype=bool)
    phase_order_score = float(np.abs(phase_lag_matrix[mask]).mean())

    return {
        "phase_order_score": phase_order_score,
        "phase_lag_matrix": phase_lag_matrix.tolist(),
    }


def _attractor_diagnostics(
    X: np.ndarray,
    burn_frac: float = 0.2,
    n_components: int = 3,
    seed: int = 0,
) -> dict:
    """
    Cluster attractors across rollouts using PCA embedding.

    :param X: trajectories (n_rollouts, T, N)
    :param burn_frac: fraction of initial timesteps to discard
    :param n_components: PCA dimensionality
    :param seed: random seed for PCA
    :return: dict with attractor diversity metrics
    """
    n_rollouts, T, N = X.shape
    burn = int(burn_frac * T)
    X_post = X[:, burn:, :]  # (n_rollouts, T-burn, N)

    # fit global PCA on all post-transient data
    X_flat = X_post.reshape(-1, N)
    pca = PCA(n_components=n_components, random_state=seed)
    pca.fit(X_flat)

    # embed each rollout and compute attractor summary (mean, radius variance)
    summaries = []
    for r in range(n_rollouts):
        proj = pca.transform(X_post[r])  # (T-burn, n_components)
        center = proj.mean(axis=0)
        radii = np.linalg.norm(proj - center, axis=1)
        mean_radius = radii.mean()
        var_radius = radii.var()
        summaries.append(np.array([*center, mean_radius, var_radius]))

    summaries = np.array(summaries)  # (n_rollouts, n_components + 2)

    # hierarchical clustering
    if n_rollouts < 2:
        return {
            "n_attractors": 1,
            "attractor_cluster_sizes": [n_rollouts],
            "attractor_separation": 0.0,
        }

    dists = pdist(summaries)
    Z = linkage(dists, method="ward")

    # choose cutoff heuristically (median distance)
    cutoff = np.median(dists) if len(dists) > 0 else 1.0
    labels = fcluster(Z, t=cutoff, criterion="distance")
    n_clusters = len(np.unique(labels))

    # cluster sizes
    cluster_sizes = [int((labels == k).sum()) for k in np.unique(labels)]

    # intra vs inter cluster distance ratio
    intra_dists = []
    inter_dists = []
    for i in range(n_rollouts):
        for j in range(i + 1, n_rollouts):
            d = np.linalg.norm(summaries[i] - summaries[j])
            if labels[i] == labels[j]:
                intra_dists.append(d)
            else:
                inter_dists.append(d)

    mean_intra = np.mean(intra_dists) if intra_dists else 0.0
    mean_inter = np.mean(inter_dists) if inter_dists else 1.0
    separation = mean_inter / (mean_intra + 1e-8)

    return {
        "n_attractors": int(n_clusters),
        "attractor_cluster_sizes": cluster_sizes,
        "attractor_separation": float(separation),
    }


def _geometry_diagnostics(
    X: np.ndarray,
    burn_frac: float = 0.2,
    seed: int = 0,
) -> dict:
    """
    Measure deviation from ellipse in 2D PCA projection.

    Uses radius variance normalized by mean radius as a proxy for curvature.

    :param X: trajectories (n_rollouts, T, N)
    :param burn_frac: fraction to discard
    :param seed: random seed
    :return: dict with geometry score
    """
    n_rollouts, T, N = X.shape
    burn = int(burn_frac * T)
    X_post = X[:, burn:, :]

    # global 2D PCA
    X_flat = X_post.reshape(-1, N)
    pca = PCA(n_components=2, random_state=seed)
    proj = pca.fit_transform(X_flat)

    center = proj.mean(axis=0)
    radii = np.linalg.norm(proj - center, axis=1)
    mean_r = radii.mean()
    std_r = radii.std()

    # coefficient of variation of radius (high = non-elliptical)
    cv_radius = std_r / (mean_r + 1e-8)

    # kurtosis of radius distribution (deviation from Gaussian)
    kurtosis = ((radii - mean_r) ** 4).mean() / (std_r ** 4 + 1e-12) - 3.0

    # combined score
    geometry_score = cv_radius + 0.1 * max(kurtosis, 0)

    return {
        "geometry_score": float(geometry_score),
        "geometry_cv_radius": float(cv_radius),
        "geometry_kurtosis": float(kurtosis),
    }


def _coupling_diagnostics(X: np.ndarray, lag: int = 2) -> dict:
    """
    Compute lagged correlation matrix from observations.

    :param X: trajectories (n_rollouts, T, N)
    :param lag: time lag in steps
    :return: dict with coupling metrics and correlation matrix
    """
    n_rollouts, T, N = X.shape

    corr_accum = np.zeros((N, N))

    for r in range(n_rollouts):
        X_t = X[r, :-lag, :]  # (T-lag, N)
        X_tp = X[r, lag:, :]  # (T-lag, N)

        # standardize
        X_t_centered = X_t - X_t.mean(axis=0, keepdims=True)
        X_tp_centered = X_tp - X_tp.mean(axis=0, keepdims=True)

        std_t = X_t_centered.std(axis=0, keepdims=True) + 1e-8
        std_tp = X_tp_centered.std(axis=0, keepdims=True) + 1e-8

        X_t_norm = X_t_centered / std_t
        X_tp_norm = X_tp_centered / std_tp

        # cross-correlation: corr[i,j] = corr(X_t[:,i], X_tp[:,j])
        corr = (X_t_norm.T @ X_tp_norm) / (T - lag)
        corr_accum += corr

    corr_matrix = corr_accum / n_rollouts

    # structure score: variance of off-diagonal entries
    mask = ~np.eye(N, dtype=bool)
    off_diag = corr_matrix[mask]
    structure_score = float(np.var(off_diag))

    # also report mean absolute off-diagonal
    mean_abs_offdiag = float(np.abs(off_diag).mean())

    return {
        "coupling_structure_score": structure_score,
        "coupling_mean_abs_offdiag": mean_abs_offdiag,
        "lagged_correlation_matrix": corr_matrix.tolist(),
    }


def evaluate_dynamics(
    trajectories: np.ndarray | torch.Tensor,
    dt: float = 1.0,
    n_memories: int | None = None,
    burn_frac: float = 0.2,
    seed: int = 0,
) -> dict:
    """
    Compute dynamical diagnostics for simulated protein trajectories.

    This function evaluates whether generated dynamics are suitable for
    AANN experiments: bounded, nonlinear, multi-attractor, with coherent
    oscillations and structured phase relationships.

    :param trajectories: shape (n_rollouts, T, N) with values in [0, 1]
    :param dt: time step for frequency calculations, defaults to 1.0
    :param n_memories: expected number of attractors (unused, for reference)
    :param burn_frac: fraction of initial timesteps to discard as transient
    :param seed: random seed for PCA
    :return: dict of scalar metrics and small arrays (JSON-serializable)
    """
    X = _to_numpy(trajectories)

    if X.ndim != 3:
        raise ValueError(
            f"Expected 3D array (n_rollouts, T, N), got shape {X.shape}")

    results = {}

    # 1. spectral coherence
    results.update(_spectral_diagnostics(X, dt))

    # 2. saturation / boundedness
    results.update(_saturation_diagnostics(X))

    # 3. phase structure
    results.update(_phase_diagnostics(X))

    # 4. attractor diversity
    results.update(_attractor_diagnostics(X, burn_frac=burn_frac, seed=seed))

    # 5. geometric nonlinearity
    results.update(_geometry_diagnostics(X, burn_frac=burn_frac, seed=seed))

    # 6. coupling structure
    results.update(_coupling_diagnostics(X))

    # add metadata
    results["n_rollouts"] = int(X.shape[0])
    results["T"] = int(X.shape[1])
    results["N"] = int(X.shape[2])
    results["dt"] = float(dt)
    if n_memories is not None:
        results["n_memories_expected"] = int(n_memories)

    return results


def summarize_diagnostics(results: dict) -> str:
    """
    Return a one-line summary of key diagnostics.

    :param results: output from evaluate_dynamics
    :return: summary string
    """
    return (
        f"freq={results['spectral_mean_freq']:.3f}±{results['spectral_freq_std']:.3f} | "
        f"sharpness={results['spectral_sharpness']:.2f} | "
        f"sat={results['mean_saturation']:.2f} | "
        f"phase={results['phase_order_score']:.2f} | "
        f"n_attr={results['n_attractors']} | "
        f"sep={results['attractor_separation']:.2f} | "
        f"geom={results['geometry_score']:.2f} | "
        f"coupling={results['coupling_structure_score']:.3f}"
    )
