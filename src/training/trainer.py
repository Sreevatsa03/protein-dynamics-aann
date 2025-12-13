from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import torch
from torch import nn
from torch.utils.data import DataLoader

from .early_stop import EarlyStopping


@dataclass
class OptimizerConfig:
    """
    Configuration for the optimizer.
    """
    lr: float = 1e-2
    momentum: float = 0.9
    weight_decay: float = 0.0


@dataclass
class TrainingConfig:
    """
    Configuration for training a model.

    :param num_epochs: number of training epochs
    :type num_epochs: int
    :param patience: number of epochs with no improvement for early stopping
    :type patience: int
    :param device: device to run training on (e.g., 'cpu' or 'cuda')
    :type device: torch.device
    :param checkpoint_path: path to save model checkpoints
    :type checkpoint_path: Path
    """

    num_epochs: int
    patience: int
    device: torch.device
    checkpoint_path: Path


class Trainer:
    """
    Trainer class to handle model training with early stopping.

    :param model: the neural network model to train
    :type model: nn.Module
    :param train_loader: DataLoader for training data
    :type train_loader: DataLoader
    :param val_loader: DataLoader for validation data
    :type val_loader: DataLoader
    :param optimizer_config: configuration for the optimizer
    :type optimizer_config: OptimizerConfig
    :param training_config: configuration for training
    :type training_config: TrainingConfig
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        optimizer_config: OptimizerConfig,
        training_config: TrainingConfig,
    ) -> None:
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.opt_cfg = optimizer_config
        self.cfg = training_config

        self.device = training_config.device
        self.model.to(self.device)

        self.criterion = nn.MSELoss()

        self.optimizer = torch.optim.SGD(
            self.model.parameters(),
            lr=self.opt_cfg.lr,
            momentum=self.opt_cfg.momentum,
            weight_decay=self.opt_cfg.weight_decay,
        )

        self.early_stopper = EarlyStopping(
            patience=self.cfg.patience,
            min_delta=0.0,
        )

        self.train_losses: List[float] = []
        self.val_losses: List[float] = []

    def train(self) -> Dict:
        """
        Train the model with early stopping.

        :return: dictionary containing best model state, loss history, and checkpoint path
        :rtype: Dict
        """
        best_val = float("inf")
        checkpoint_path = Path(self.cfg.checkpoint_path)
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

        for epoch in range(self.cfg.num_epochs):
            train_loss = self.train_one_epoch()
            val_loss = self.evaluate()

            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)

            # check for improvement
            if val_loss < best_val:
                best_val = val_loss
                self.save_checkpoint(checkpoint_path)

            # early stopping check
            if self.early_stopper.step(val_loss):
                break

        # load best model
        self.load_checkpoint(checkpoint_path)

        return {
            "model": self.model,
            "train_losses": self.train_losses,
            "val_losses": self.val_losses,
            "best_val_loss": best_val,
            "checkpoint": str(checkpoint_path),
        }

    def train_one_epoch(self) -> float:
        """
        Train for a single epoch.

        :return: average training loss for the epoch
        :rtype: float
        """
        self.model.train()
        epoch_loss = 0.0
        n_batches = 0

        for x, y in self.train_loader:
            x = x.to(self.device)
            y = y.to(self.device)

            # forward
            y_hat = self.model(x)
            loss = self.criterion(y_hat, y)

            # backward
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        return epoch_loss / n_batches

    def evaluate(self) -> float:
        """
        Compute mean validation loss.

        :return: average validation loss
        :rtype: float
        """
        self.model.eval()
        epoch_loss = 0.0
        n_batches = 0

        with torch.no_grad():
            for x, y in self.val_loader:
                x = x.to(self.device)
                y = y.to(self.device)
                y_hat = self.model(x)
                loss = self.criterion(y_hat, y)
                epoch_loss += loss.item()
                n_batches += 1

        return epoch_loss / n_batches

    def save_checkpoint(self, path: Path) -> None:
        """
        Save model checkpoint.

        :param path: path to save the checkpoint
        :type path: Path
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.model.state_dict(), path)

    def load_checkpoint(self, path: Path) -> None:
        """
        Load model weights from a checkpoint.

        :param path: path to the checkpoint
        :type path: Path
        """
        state_dict = torch.load(
            path, map_location=self.device, weights_only=True)
        self.model.load_state_dict(state_dict)
