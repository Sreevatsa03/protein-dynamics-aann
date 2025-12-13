from __future__ import annotations
from typing import Union

import numpy as np
import torch


ArrayLike = Union[np.ndarray, torch.Tensor]


def _to_numpy(x: ArrayLike) -> np.ndarray:
    """
    Convert a torch tensor or numpy array to a numpy array on cpu.
    """
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return x


def mse(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    """
    Compute mean squared error over all samples and dimensions.
    """
    yt = _to_numpy(y_true)
    yp = _to_numpy(y_pred)
    return float(np.mean((yp - yt) ** 2))


def per_protein_mse(y_true: ArrayLike, y_pred: ArrayLike) -> np.ndarray:
    """
    Compute MSE per protein (dimension).\\
    Returns an array of shape (d,).
    """
    yt = _to_numpy(y_true)
    yp = _to_numpy(y_pred)
    return np.mean((yp - yt) ** 2, axis=0)


def per_protein_correlation(y_true: ArrayLike, y_pred: ArrayLike, eps: float = 1e-12) -> np.ndarray:
    """
    Compute Pearson correlation per protein across samples.\\
    Returns an array of shape (d,).\\
    If a dimension has near-zero variance, its correlation is set to nan.
    """
    yt = _to_numpy(y_true)
    yp = _to_numpy(y_pred)

    yt_centered = yt - yt.mean(axis=0, keepdims=True)
    yp_centered = yp - yp.mean(axis=0, keepdims=True)

    num = np.sum(yt_centered * yp_centered, axis=0)
    den = np.sqrt(np.sum(yt_centered ** 2, axis=0)
                  * np.sum(yp_centered ** 2, axis=0))

    corr = np.full(yt.shape[1], np.nan, dtype=float)
    valid = den > eps
    corr[valid] = num[valid] / den[valid]
    return corr


def per_protein_r2(y_true: ArrayLike, y_pred: ArrayLike, eps: float = 1e-12) -> np.ndarray:
    """
    Compute R^2 per protein across samples.\\
    Returns an array of shape (d,).\\
    If a dimension has near-zero variance, its R^2 is set to nan.
    """
    yt = _to_numpy(y_true)
    yp = _to_numpy(y_pred)

    ss_res = np.sum((yp - yt) ** 2, axis=0)
    yt_mean = np.mean(yt, axis=0, keepdims=True)
    ss_tot = np.sum((yt - yt_mean) ** 2, axis=0)

    r2 = np.full(yt.shape[1], np.nan, dtype=float)
    valid = ss_tot > eps
    r2[valid] = 1.0 - (ss_res[valid] / ss_tot[valid])
    return r2


def rollout_mse(true_traj: ArrayLike, pred_traj: ArrayLike) -> float:
    """
    Compute mean squared error over an entire rollout trajectory.
    """
    yt = _to_numpy(true_traj)
    yp = _to_numpy(pred_traj)
    return float(np.mean((yp - yt) ** 2))


def rollout_mse_vs_time(true_traj: ArrayLike, pred_traj: ArrayLike) -> np.ndarray:
    """
    Compute MSE at each timestep of a rollout.
    Returns an array of shape (T,).
    """
    yt = _to_numpy(true_traj)
    yp = _to_numpy(pred_traj)
    return np.mean((yp - yt) ** 2, axis=1)
