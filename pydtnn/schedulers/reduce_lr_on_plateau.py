"""
Reduce learning rate when a metric has stopped improving.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np

from pydtnn.schedulers.scheduler_with_loss_or_metric import SchedulerWithLossOrMetric

__all__ = ("ReduceLROnPlateau",)

logger = logging.getLogger(__name__)


if TYPE_CHECKING:
    from pydtnn.model import Model


class ReduceLROnPlateau(SchedulerWithLossOrMetric):
    """
    ReduceLROnPlateau LRScheduler
    """

    def __init__(self, loss_or_metric: str = "", factor=0.1, patience=5, min_lr=0, verbose=True):
        """
        Initialize the ReduceLROnPlateau scheduler.

        Args:
            loss_or_metric: The name of the loss or metric to monitor.
            factor: Factor by which the learning rate will be reduced.
            patience: Number of epochs with no improvement after which learning rate will be reduced.
            min_lr: A lower bound on the learning rate.
            verbose: Whether to print updates to the logger.
        """
        # NOTE: loss_or_metric default value is "val_accuracy" in Parser.
        super().__init__(loss_or_metric, verbose)
        self.factor = factor
        self.patience = patience
        self.min_lr = min_lr
        self.best_epoch: int = 0
        self.best_loss: float = np.inf * {True: -1, False: 1}["accuracy" in self.loss_or_metric]

    def on_epoch_end(self, train_loss: np.ndarray, val_loss: np.ndarray) -> None:
        """
        Update the learning rate at the end of an epoch if the monitored metric has plateaued.

        Args:
            train_loss: Array of training losses.
            val_loss: Array of validation losses.
        """
        idx = self._get_idx()
        self.epoch_count += 1
        loss = val_loss if self.is_val_metric else train_loss
        if self.compare(loss[idx], self.best_loss):
            self.best_loss = loss[idx]
            self.best_epoch = self.epoch_count
        elif (
            self.epoch_count - self.best_epoch >= self.patience
            and self.model.optimizer.learning_rate * self.factor >= self.min_lr
        ):
            self.model.optimizer.learning_rate *= self.factor
            self.best_epoch = self.epoch_count
            self.log(
                f"Metric {self.loss_or_metric} did not improve for {
                    self.patience
                } epochs, setting learning rate to {self.model.optimizer.learning_rate:.8f}."
            )

    @classmethod
    def from_model(cls, model: Model) -> ReduceLROnPlateau:
        """
        Create a ReduceLROnPlateau instance from a model configuration.

        Args:
            model: The model instance containing scheduler parameters.

        Returns:
            An instance of ReduceLROnPlateau.
        """
        return ReduceLROnPlateau(
            model.reduce_lr_on_plateau_metric,
            model.reduce_lr_on_plateau_factor,
            model.reduce_lr_on_plateau_patience,
            model.reduce_lr_on_plateau_min_lr,
        )
