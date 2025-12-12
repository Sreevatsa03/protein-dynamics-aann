import torch
from torch.distributions import LogNormal


def build_gt_W(
    S: torch.Tensor,
    mu: float = -1.2,
    sigma: float = 0.6,
    target_radius: float = 0.9
) -> torch.Tensor:
    """
    Construct the effective interaction weight matrix W_eff.

    :param S: signed adjacency matrix in {-1, 0, +1}^{d x d}
    :type S: torch.Tensor
    :param mu: mean of the underlying normal distribution for log-normal sampling, defaults to -1.2
    :type mu: float, optional
    :param sigma: standard deviation of the underlying normal distribution for log-normal sampling, defaults to 0.6
    :type sigma: float, optional
    :param target_radius: desired spectral radius of W_eff, defaults to 0.9
    :type target_radius: float, optional
    :return: effective interaction weight matrix W_eff
    :rtype: torch.Tensor
    """
    d = S.shape[0]

    # sample positive magnitudes
    dist = LogNormal(mu, sigma)
    magnitudes = dist.sample((d, d))

    # apply signs from S
    W_eff = S * magnitudes

    # compute current spectral radius
    eigenvalues = torch.linalg.eigvals(W_eff)
    rho = eigenvalues.abs().max().real

    if rho < 1e-8:
        raise ValueError("Spectral radius too small, mask S may be empty.")

    # rescale to target_radius
    W_eff = W_eff * (target_radius / rho)
    return W_eff
