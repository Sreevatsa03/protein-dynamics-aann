from __future__ import annotations
from typing import Union

import numpy as np
import torch


ArrayLike = Union[np.ndarray, torch.Tensor]


def _to_numpy(x: ArrayLike) -> np.ndarray:
    """Convert a tensor-like input to a NumPy array on cpu.

    :param x: input array or tensor
    :type x: ArrayLike
    :return: array converted to NumPy on cpu
    :rtype: np.ndarray
    """
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return x


def mse(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    """Compute mean squared error over all samples and dimensions.

    :param y_true: ground-truth targets
    :type y_true: ArrayLike
    :param y_pred: predicted values
    :type y_pred: ArrayLike
    :return: mean squared error as a scalar
    :rtype: float
    """
    yt = _to_numpy(y_true)
    yp = _to_numpy(y_pred)
    return float(np.mean((yp - yt) ** 2))


def per_protein_mse(y_true: ArrayLike, y_pred: ArrayLike) -> np.ndarray:
    """Compute mean squared error per protein (dimension).

    :param y_true: ground-truth targets of shape (..., d)
    :type y_true: ArrayLike
    :param y_pred: predicted values of shape (..., d)
    :type y_pred: ArrayLike
    :return: per-protein MSE of shape (d,)
    :rtype: np.ndarray
    """
    yt = _to_numpy(y_true)
    yp = _to_numpy(y_pred)
    return np.mean((yp - yt) ** 2, axis=0)


def per_protein_correlation(y_true: ArrayLike, y_pred: ArrayLike, eps: float = 1e-12) -> np.ndarray:
    """Compute Pearson correlation per protein across samples.

    if a dimension has near-zero variance, its correlation is set to nan.

    :param y_true: ground-truth targets of shape (n_samples, d)
    :type y_true: ArrayLike
    :param y_pred: predicted values of shape (n_samples, d)
    :type y_pred: ArrayLike
    :param eps: numerical threshold for variance, defaults to 1e-12
    :type eps: float, optional
    :return: per-protein Pearson correlations of shape (d,)
    :rtype: np.ndarray
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
    """Compute coefficient of determination (R^2) per protein.

    if a dimension has near-zero variance, its R^2 is set to nan.

    :param y_true: ground-truth targets of shape (n_samples, d)
    :type y_true: ArrayLike
    :param y_pred: predicted values of shape (n_samples, d)
    :type y_pred: ArrayLike
    :param eps: numerical threshold for variance, defaults to 1e-12
    :type eps: float, optional
    :return: per-protein R^2 scores of shape (d,)
    :rtype: np.ndarray
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
    """Compute mean squared error over an entire rollout trajectory.

    :param true_traj: ground-truth rollout of shape (T, d) or (batch, T, d)
    :type true_traj: ArrayLike
    :param pred_traj: predicted rollout with same shape as ``true_traj``
    :type pred_traj: ArrayLike
    :return: mean squared error over all timesteps and dimensions
    :rtype: float
    """
    yt = _to_numpy(true_traj)
    yp = _to_numpy(pred_traj)
    return float(np.mean((yp - yt) ** 2))


def rollout_mse_vs_time(true_traj: ArrayLike, pred_traj: ArrayLike) -> np.ndarray:
    """Compute MSE at each timestep of a rollout.

    :param true_traj: ground-truth rollout of shape (T, d) or (batch, T, d)
    :type true_traj: ArrayLike
    :param pred_traj: predicted rollout with same shape as ``true_traj``
    :type pred_traj: ArrayLike
    :return: mean squared error per timestep of shape (T,)
    :rtype: np.ndarray
    """
    yt = _to_numpy(true_traj)
    yp = _to_numpy(pred_traj)
    return np.mean((yp - yt) ** 2, axis=-1)


def open_loop_rollout(model, x0: torch.Tensor, T: int) -> torch.Tensor:
    """
    Perform open-loop rollout from initial condition.

    :param model: trained AANN model (LinearAANN or SigmoidAANN)
    :param x0: initial state of shape (state_dim,) or (batch, state_dim)
    :param T: number of steps to roll out
    :return: trajectory of shape (T, state_dim) or (batch, T, state_dim)
    """
    with torch.no_grad():
        # handle single vs batched input
        if x0.dim() == 1:
            x0 = x0.unsqueeze(0)
            squeeze_output = True
        else:
            squeeze_output = False

        traj = [x0]
        x = x0
        for _ in range(T - 1):
            x = model(x)
            traj.append(x)

        result = torch.stack(traj, dim=1)  # (batch, T, state_dim)

        if squeeze_output:
            result = result.squeeze(0)  # (T, state_dim)

        return result
