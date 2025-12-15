from __future__ import annotations

import torch


def spectral_radius(W: torch.Tensor) -> float:
    """compute the spectral radius of ``W``.

    :param W: square matrix
    :type W: torch.Tensor
    :return: spectral radius of ``W``
    :rtype: float
    """
    # compute eigenvalues on cpu for speed/stability
    eigvals = torch.linalg.eigvals(W.cpu())
    return eigvals.abs().max().item()


def scale_to_spectral_radius(W: torch.Tensor, target_radius: float) -> torch.Tensor:
    """rescale ``W`` so that its spectral radius becomes ``target_radius``.

    :param W: weight matrix to rescale
    :type W: torch.Tensor
    :param target_radius: desired spectral radius
    :type target_radius: float
    :return: rescaled matrix with spectral radius approximately equal to ``target_radius``
    :rtype: torch.Tensor
    """
    rho = spectral_radius(W)
    if rho == 0:
        return W
    scale = target_radius / rho
    return W * scale
