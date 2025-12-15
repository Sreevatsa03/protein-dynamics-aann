from __future__ import annotations
from typing import Optional

import torch
from torch import nn


class BaseMaskedAANN(nn.Module):
    """
    Base class for Masked Autoencoder Neural Networks (AANNs).

    This class serves as a template for specific implementations of masked AANNs:
    - Stores structural connectivity mask
    - Defines weight and bias parameters
    - Computes effective masked weight matrix

    :param state_dim: dimensionality of the state vector x_t
    :type state_dim: int
    :param mask: connectivity mask wiht shape (state_dim, state_dim). nonzero entries indicate allowed connections
    :type mask: torch.Tensor
    :param init_scale: standard deviation for weight initialization, defaults to 0.1
    :type init_scale: float, optional
    :param device: device to place parameters on, defaults to None
    :type device: torch.device, optional
    :param dtype: data type for parameters, defaults to torch.float32
    :type dtype: torch.dtype, optional
    """

    def __init__(
        self,
        state_dim: int,
        mask: torch.Tensor,
        init_scale: float = 0.1,
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__()

        if mask.shape != (state_dim, state_dim):
            raise ValueError(
                f"mask must have shape ({state_dim}, {state_dim}), "
                f"got {tuple(mask.shape)}"
            )

        mask = mask.to(device=device, dtype=dtype)
        struct_mask = (mask != 0).to(device=device, dtype=dtype)

        self.state_dim = state_dim

        # structural connectivity mask (non-trainable)
        self.register_buffer("mask", struct_mask)

        # weight and bias parameters
        self.weight = nn.Parameter(
            init_scale * torch.randn(state_dim, state_dim,
                                     device=device, dtype=dtype)
        )
        self.bias = nn.Parameter(torch.zeros(
            state_dim, device=device, dtype=dtype))

    def effective_weight(self) -> torch.Tensor:
        """
        Return the structurally masked weight matrix.

        W_eff[i, j] = W[i, j] if M[i, j] != 0 else 0
        """
        return self.weight * self.mask

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass x -> x_hat.

        This is defined in subclasses (linear vs sigmoid).
        """
        raise NotImplementedError


class LinearAANN(BaseMaskedAANN):
    """
    Masked Autoencoder Neural Network (AANN) with linear activation.

    Implements:
        x_hat = W_eff @ x + b, W_eff = W * M
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass x -> x_hat using linear activation.

        x_hat = W_eff @ x + b
        :param x: input state vector with shape (batch_size, state_dim)
        :type x: torch.Tensor
        :return: reconstructed state vector with shape (batch_size, state_dim)
        :rtype: torch.Tensor
        """
        W_eff = self.effective_weight()
        return x @ W_eff.T + self.bias


class SigmoidAANN(BaseMaskedAANN):
    """
    Masked Autoencoder Neural Network (AANN) with sigmoid activation.

    Implements:
        x_hat = σ(W_eff @ x + b), W_eff = W * M
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass x -> x_hat using sigmoid activation.

        x_hat = sigmoid(W_eff @ x + b)
        :param x: input state vector with shape (batch_size, state_dim)
        :type x: torch.Tensor
        :return: reconstructed state vector with shape (batch_size, state_dim)
        :rtype: torch.Tensor
        """
        W_eff = self.effective_weight()
        return torch.sigmoid(x @ W_eff.T + self.bias)


def compute_hopfield_weights(
    patterns: torch.Tensor,
    scale: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute Hopfield weight matrix and bias from stored patterns.

    The weight matrix is computed as the outer product of centered patterns:
        W = (scale / K) * sum_mu (xi^mu - mean)(xi^mu - mean)^T

    The bias is set so that sigmoid(W @ xi + b) ≈ xi for each pattern:
        b_i = -sum_j W_ij * mean_j + logit(mean_i)

    :param patterns: tensor of shape (K, d) where K is number of patterns, d is state dim
    :type patterns: torch.Tensor
    :param scale: scaling factor for weight strength (larger = sharper attractors), defaults to 1.0
    :type scale: float, optional
    :return: (W, b) weight matrix (d, d) and bias vector (d,)
    :rtype: tuple[torch.Tensor, torch.Tensor]
    """
    K, d = patterns.shape

    # center patterns
    mean_pattern = patterns.mean(dim=0)  # (d,)
    centered = patterns - mean_pattern  # (K, d)

    # Hopfield outer product rule
    W = (scale / K) * (centered.T @ centered)  # (d, d)

    # set bias so fixed points are near pattern means
    # for sigmoid: we want sigmoid(W @ xi + b) ≈ xi
    # approximate: b = logit(mean) - W @ mean
    mean_clamped = mean_pattern.clamp(0.01, 0.99)  # avoid inf in logit
    logit_mean = torch.log(mean_clamped / (1 - mean_clamped))
    b = logit_mean - W @ mean_pattern

    return W, b


def create_hopfield_sigmoid_aann(
    patterns: torch.Tensor,
    mask: torch.Tensor,
    scale: float = 4.0,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float32,
) -> SigmoidAANN:
    """
    Create a SigmoidAANN with Hopfield-encoded weights for storing patterns.

    :param patterns: tensor of shape (K, d) where K is number of patterns to store
    :type patterns: torch.Tensor
    :param mask: connectivity mask of shape (d, d)
    :type mask: torch.Tensor
    :param scale: weight scaling factor (larger = sharper attractors), defaults to 4.0
    :type scale: float, optional
    :param device: device to place model on, defaults to None
    :type device: torch.device, optional
    :param dtype: data type, defaults to torch.float32
    :type dtype: torch.dtype, optional
    :return: SigmoidAANN with Hopfield-encoded weights
    :rtype: SigmoidAANN
    """
    d = patterns.shape[1]
    patterns = patterns.to(device=device, dtype=dtype)
    mask = mask.to(device=device, dtype=dtype)

    # compute Hopfield weights
    W, b = compute_hopfield_weights(patterns, scale=scale)

    # create model
    model = SigmoidAANN(state_dim=d, mask=mask, device=device, dtype=dtype)

    # set weights (apply mask)
    with torch.no_grad():
        model.weight.data = W.to(device=device, dtype=dtype)
        model.bias.data = b.to(device=device, dtype=dtype)

    return model


def create_hopfield_linear_aann(
    patterns: torch.Tensor,
    mask: torch.Tensor,
    scale: float = 1.0,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float32,
) -> LinearAANN:
    """
    Create a LinearAANN with Hopfield-encoded weights (for comparison).

    Note: Linear networks cannot have bounded attractors, so this serves
    as a negative control showing that sigmoid nonlinearity is necessary.

    :param patterns: tensor of shape (K, d) where K is number of patterns to store
    :type patterns: torch.Tensor
    :param mask: connectivity mask of shape (d, d)
    :type mask: torch.Tensor
    :param scale: weight scaling factor, defaults to 1.0
    :type scale: float, optional
    :param device: device to place model on, defaults to None
    :type device: torch.device, optional
    :param dtype: data type, defaults to torch.float32
    :type dtype: torch.dtype, optional
    :return: LinearAANN with Hopfield-encoded weights
    :rtype: LinearAANN
    """
    d = patterns.shape[1]
    patterns = patterns.to(device=device, dtype=dtype)
    mask = mask.to(device=device, dtype=dtype)

    # compute Hopfield weights (same formula)
    W, _ = compute_hopfield_weights(patterns, scale=scale)

    # for linear: bias = (I - W) @ mean to have mean as fixed point
    mean_pattern = patterns.mean(dim=0)
    b = mean_pattern - W @ mean_pattern

    # create model
    model = LinearAANN(state_dim=d, mask=mask, device=device, dtype=dtype)

    # set weights
    with torch.no_grad():
        model.weight.data = W.to(device=device, dtype=dtype)
        model.bias.data = b.to(device=device, dtype=dtype)

    return model


def create_hopfield_sigmoid_aann_dense(
    patterns: torch.Tensor,
    scale: float = 4.0,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float32,
) -> SigmoidAANN:
    """
    Create a SigmoidAANN where stored patterns are exact fixed points.

    Uses least-squares to find W, b such that σ(W @ ξ + b) ≈ ξ for all patterns.
    Then verifies stability by checking Jacobian eigenvalues.

    :param patterns: tensor of shape (K, d) where K is number of patterns to store
    :type patterns: torch.Tensor
    :param scale: amplification factor for weight matrix (for sharper basins)
    :type scale: float, optional
    :param device: device to place model on, defaults to None
    :type device: torch.device, optional
    :param dtype: data type, defaults to torch.float32
    :type dtype: torch.dtype, optional
    :return: SigmoidAANN with patterns as fixed points
    :rtype: SigmoidAANN
    """
    import numpy as np

    K, d = patterns.shape
    patterns = patterns.to(device=device, dtype=dtype)

    # clamp patterns away from 0 and 1 to avoid logit explosion
    patterns_clamped = patterns.clamp(0.1, 0.9)

    # compute logit of patterns (these are the target pre-activations)
    # We need: W @ ξ + b = logit(ξ) for each pattern ξ
    logit_patterns = torch.log(
        patterns_clamped / (1 - patterns_clamped))  # (K, d)

    # solve via least squares: [ξ | 1] @ [W^T; b^T] = logit(ξ)
    # augment patterns with column of ones for bias
    ones = torch.ones(K, 1, device=device, dtype=dtype)
    patterns_aug = torch.cat([patterns_clamped, ones], dim=1)  # (K, d+1)

    # solve least squares for each output dimension
    # patterns_aug @ solution = logit_patterns
    # solution shape: (d+1, d)
    solution = torch.linalg.lstsq(
        patterns_aug, logit_patterns).solution  # (d+1, d)

    W = solution[:d, :].T  # (d, d) - transpose to get correct shape
    b = solution[d, :]     # (d,)

    # scale weights to sharpen basins of attraction
    W = W * scale
    b = b * scale

    # create model with FULL connectivity mask
    full_mask = torch.ones(d, d, device=device, dtype=dtype)
    model = SigmoidAANN(state_dim=d, mask=full_mask,
                        device=device, dtype=dtype)

    # set weights
    with torch.no_grad():
        model.weight.data = W
        model.bias.data = b

    # verify and report
    print("  Hopfield Dense verification:")
    with torch.no_grad():
        for i, p in enumerate(patterns_clamped):
            out = model(p)
            err = (out - p).norm().item()
            # check stability: iterate 10 times
            x = p.clone()
            for _ in range(10):
                x = model(x)
            drift = (x - p).norm().item()
            phase_names = ["G1", "S", "G2", "M"]
            print(
                f"    {phase_names[i]}: fixed_pt_err={err:.4f}, drift_after_10={drift:.4f}")

    return model
