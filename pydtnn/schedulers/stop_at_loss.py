"""Module for the StopAtLoss scheduler, which terminates training based on loss or metric thresholds."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from numpy import ndarray

from pydtnn.schedulers.scheduler_with_loss_or_metric import SchedulerWithLossOrMetric

__all__ = ("StopAtLoss",)

logger = logging.getLogger(__name__)


if TYPE_CHECKING:
    from pydtnn.model import Model


class StopAtLoss(SchedulerWithLossOrMetric):
    """Scheduler that stops training when a specific loss or metric reaches a threshold."""

    def __init__(
        self, loss_or_metric: str = "", threshold_value: float = 0.0, verbose: bool = True
    ) -> None:
        """
        Initialize the StopAtLoss scheduler.

        Args:
            loss_or_metric: The name of the loss or metric to monitor.
            threshold_value: The value at which to stop training.
            verbose: Whether to log status updates.
        """
        # NOTE: loss_or_metric default value is "val_accuracy" in Parser.
        super().__init__(loss_or_metric, verbose)
        self.threshold_value = threshold_value

    def on_epoch_end(self, train_loss: ndarray, val_loss: ndarray) -> None:
        """
        Check if the monitored metric has reached the threshold at the end of an epoch.

        Args:
            train_loss: Array of training losses.
            val_loss: Array of validation losses.
        """
        idx = self._get_idx()
        self.epoch_count += 1
        loss = val_loss if self.is_val_metric else train_loss
        if self.compare(loss[idx], self.threshold_value):
            self.stop_training = True
            self.log(
                f"Metric '{self.loss_or_metric}' reached threshold value {
                    self.threshold_value
                }, stop training."
            )

    @classmethod
    def from_model(cls, model: Model) -> StopAtLoss:
        """
        Create a StopAtLoss instance from a model configuration.

        Args:
            model: The model instance containing stop criteria.

        Returns:
            An instance of StopAtLoss.
        """
        return StopAtLoss(model.stop_at_loss_metric, model.stop_at_loss_threshold)
