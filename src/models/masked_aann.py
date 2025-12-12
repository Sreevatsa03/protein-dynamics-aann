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
