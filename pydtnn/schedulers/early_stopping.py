"""
Early stopping module for PyDTNN.

This module provides the EarlyStopping class, which monitors model performance
during training and terminates the process if no improvement is observed
within a specified number of epochs.
"""

from __future__ import annotations

import logging
import operator
from typing import TYPE_CHECKING

import numpy as np

from pydtnn import timestamp
from pydtnn.schedulers.scheduler_with_loss_or_metric import SchedulerWithLossOrMetric

__all__ = ("EarlyStopping",)

logger = logging.getLogger(__name__)


if TYPE_CHECKING:
    from pydtnn.model import Model


class EarlyStopping(SchedulerWithLossOrMetric):
    """
    Early stopping scheduler to terminate training when a monitored metric stops improving.
    """

    model: Model

    def __init__(self, loss_or_metric="", patience=10, minimize=True, verbose=True):
        """
        Initialize the EarlyStopping scheduler.

        Args:
            loss_or_metric: Name of the metric to monitor.
            patience: Number of epochs to wait for improvement before stopping.
            minimize: Whether the metric should be minimized.
            verbose: Whether to log status updates.
        """
        super().__init__(loss_or_metric, verbose)
        self.patience = patience
        self.minimize = minimize
        self.best_epoch: int = 0
        self.best_loss_or_metric: float = np.inf * {True: -1, False: 1}[not self.minimize]
        self.best_weights_filename: str | None = None
        self.compare = operator.lt if self.minimize else operator.gt

    def on_epoch_end(
        self, train_loss_or_metrics: np.ndarray, val_loss_or_metrics: np.ndarray
    ) -> None:
        """
        Check if the monitored metric has improved at the end of an epoch.

        Args:
            train_loss_or_metrics: Array of training metrics.
            val_loss_or_metrics: Array of validation metrics.
        """
        idx = self._get_idx()
        self.epoch_count += 1
        loss_or_metrics = val_loss_or_metrics if self.is_val_metric else train_loss_or_metrics

        if self.compare(loss_or_metrics[idx], self.best_loss_or_metric):
            self.best_loss_or_metric = loss_or_metrics[idx]
            self.best_epoch = self.epoch_count
            # Save weights + bias
            if not self.best_weights_filename:
                self.best_weights_filename = f"./model-{self.model.model_name}-weights-rank_{
                    self.model.comm_rank
                }-{timestamp}.npz"
            self.model.save_model_state(self.best_weights_filename, compress=False)
        elif (self.epoch_count - self.best_epoch) >= self.patience:
            self.stop_training = True
            # Restore weights + bias
            assert self.best_weights_filename
            self.model.load_model_state(self.best_weights_filename)
            self.log(
                f"Metric '{self.loss_or_metric}' did not improve for {
                    self.patience
                } epochs, stop training."
            )
        # else: do nothing.

    @classmethod
    def from_model(cls, model: Model) -> EarlyStopping:
        """
        Create an EarlyStopping instance from a model configuration.

        Args:
            model: The model instance containing early stopping parameters.

        Returns:
            An initialized EarlyStopping scheduler.
        """
        return EarlyStopping(
            model.early_stopping_metric,
            model.early_stopping_patience,
            model.early_stopping_minimize,
        )
