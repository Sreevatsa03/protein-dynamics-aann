from __future__ import annotations

import torch
from torch import nn

from src.models.masked_aann import LinearAANN, SigmoidAANN


def hebbian_update(
    model: nn.Module,
    x: torch.Tensor,
    target: torch.Tensor,
    lr: float = 1e-3,
    update_bias: bool = True,
) -> float:
    """
    Perform a single Hebbian weight update on an AANN model.

    Computes:
        output = model(x)
        E = target - output
        ΔW = lr * outer(E, x)  (averaged over batch)
        W += ΔW * mask

    :param model: LinearAANN or SigmoidAANN model
    :param x: input states, shape (batch, state_dim)
    :param target: target states, shape (batch, state_dim)
    :param lr: learning rate β, defaults to 1e-3
    :param update_bias: whether to update bias, defaults to True
    :return: batch MSE loss (for logging)
    """
    with torch.no_grad():
        # forward pass
        output = model(x)

        # compute per-neuron error: E = target - output
        # shape: (batch, state_dim)
        E = target - output

        # compute MSE for logging
        mse = (E ** 2).mean().item()

        # Hebbian weight update: ΔW_ij = β * E_i * x_j
        # W[i, j] connects x[j] -> output[i]
        # ΔW = lr * E.T @ x / batch_size (outer product averaged)
        batch_size = x.shape[0]

        # E: (batch, state_dim) -> need (state_dim, batch)
        # x: (batch, state_dim)
        # ΔW: (state_dim, state_dim) where ΔW[i,j] = sum_b E[b,i] * x[b,j] / batch
        delta_W = (E.T @ x) / batch_size  # (state_dim, state_dim)

        # apply learning rate
        delta_W = lr * delta_W

        # update weights with mask enforcement
        model.weight.data += delta_W
        model.weight.data *= model.mask  # enforce structural connectivity

        # stabilize weights to avoid sigmoid saturation
        max_norm = 1.0
        weight_norm = model.weight.data.norm()
        if weight_norm > max_norm:
            model.weight.data *= max_norm / weight_norm

        # update bias if requested
        if update_bias:
            # Δb_i = β * mean(E_i)
            delta_b = lr * E.mean(dim=0)
            model.bias.data += delta_b

    return mse


def hebbian_train_epoch(
    model: nn.Module,
    train_loader: torch.utils.data.DataLoader,
    lr: float = 1e-3,
    update_bias: bool = True,
) -> float:
    """
    Train one epoch using Hebbian learning.

    :param model: LinearAANN or SigmoidAANN model
    :param train_loader: DataLoader yielding (x_t, x_{t+1}) pairs
    :param lr: learning rate β
    :param update_bias: whether to update bias
    :return: average MSE over epoch
    """
    model.eval()  # no dropout/batchnorm, but keeps model in consistent state

    total_loss = 0.0
    n_batches = 0

    for x, target in train_loader:
        mse = hebbian_update(model, x, target, lr=lr, update_bias=update_bias)
        total_loss += mse
        n_batches += 1

    return total_loss / n_batches if n_batches > 0 else 0.0


def evaluate_model(
    model: nn.Module,
    data_loader: torch.utils.data.DataLoader,
) -> dict:
    """
    Evaluate model on a dataset (teacher-forced).

    :param model: LinearAANN or SigmoidAANN
    :param data_loader: DataLoader yielding (x, target) pairs
    :return: dict with mse, per_protein_r2
    """
    model.eval()

    all_targets = []
    all_outputs = []

    with torch.no_grad():
        for x, target in data_loader:
            output = model(x)
            all_targets.append(target)
            all_outputs.append(output)

    targets = torch.cat(all_targets, dim=0)
    outputs = torch.cat(all_outputs, dim=0)

    # MSE
    mse = ((outputs - targets) ** 2).mean().item()

    # per-protein R2
    ss_res = ((outputs - targets) ** 2).sum(dim=0)
    ss_tot = ((targets - targets.mean(dim=0, keepdim=True)) ** 2).sum(dim=0)
    r2 = 1.0 - ss_res / (ss_tot + 1e-12)
    r2 = r2.numpy()

    return {
        "mse": mse,
        "per_protein_r2": r2.tolist(),
        "mean_r2": float(r2.mean()),
    }


class HebbianTrainer:
    """
    Trainer for Hebbian learning with early stopping.

    :param model: LinearAANN or SigmoidAANN
    :param train_loader: training DataLoader
    :param val_loader: validation DataLoader
    :param lr: learning rate β
    :param update_bias: whether to update bias
    :param patience: early stopping patience
    :param min_delta: minimum improvement for early stopping
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: torch.utils.data.DataLoader,
        val_loader: torch.utils.data.DataLoader,
        lr: float = 1e-3,
        update_bias: bool = True,
        patience: int = 50,
        min_delta: float = 1e-6,
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.lr = lr
        self.update_bias = update_bias
        self.patience = patience
        self.min_delta = min_delta

        self.train_losses: list[float] = []
        self.val_losses: list[float] = []
        self.val_r2s: list[float] = []

    def train(self, num_epochs: int = 500) -> dict:
        """
        Train the model for multiple epochs with early stopping.

        :param num_epochs: maximum number of epochs
        :return: dict with training history and best metrics
        """
        best_val_loss = float("inf")
        best_state = None
        epochs_without_improvement = 0

        for epoch in range(num_epochs):
            # train one epoch
            train_loss = hebbian_train_epoch(
                self.model,
                self.train_loader,
                lr=self.lr,
                update_bias=self.update_bias,
            )
            self.train_losses.append(train_loss)

            # evaluate on validation set
            val_metrics = evaluate_model(self.model, self.val_loader)
            val_loss = val_metrics["mse"]
            val_r2 = val_metrics["mean_r2"]

            self.val_losses.append(val_loss)
            self.val_r2s.append(val_r2)

            # check for improvement
            if val_loss < best_val_loss - self.min_delta:
                best_val_loss = val_loss
                best_state = {k: v.clone()
                              for k, v in self.model.state_dict().items()}
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1

            # early stopping
            if epochs_without_improvement >= self.patience:
                print(f"Early stopping at epoch {epoch + 1}")
                break

        # restore best model
        if best_state is not None:
            self.model.load_state_dict(best_state)

        return {
            "train_losses": self.train_losses,
            "val_losses": self.val_losses,
            "val_r2s": self.val_r2s,
            "best_val_loss": best_val_loss,
            "epochs_trained": len(self.train_losses),
        }
