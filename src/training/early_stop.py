from __future__ import annotations
from dataclasses import dataclass
from typing import Optional


@dataclass
class EarlyStopping:
    """
    Simple early stopping mechanism tracking validation loss improvements.

    :param patience: number of epochs with no improvement after which training will be stopped, defaults to 10
    :type patience: int, optional
    :param min_delta: minimum change in the monitored metric to qualify as an improvement, defaults to 0.0
    :type min_delta: float, optional
    """
    patience: int = 10
    min_delta: float = 0.0

    def __post_init__(self):
        self.best_loss: float = float("inf")
        self.counter: int = 0
        self.early_stop: bool = False

    def step(self, val_loss: float) -> bool:
        """
        Check if validation loss has improved and update early stopping state.

        :param val_loss: current validation loss
        :type val_loss: float
        :return: whether training should be stopped early
        :rtype: bool
        """
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

        return self.early_stop
