"""Module for the ReduceLREveryNEpochs learning rate scheduler."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from numpy import ndarray

from pydtnn.schedulers.abstract.scheduler import Scheduler

__all__ = ("ReduceLREveryNEpochs",)

logger = logging.getLogger(__name__)


if TYPE_CHECKING:
    from pydtnn.model import Model


class ReduceLREveryNEpochs(Scheduler):
    """ReduceLREveryNEpochs LRScheduler"""

    def __init__(
        self, factor: float = 0.1, nepochs: int = 5, min_lr: float = 0.0, verbose: bool = True
    ) -> None:
        """
        Initialize the scheduler.

        Args:
            factor: Multiplicative factor of learning rate decay.
            nepochs: Number of epochs to wait before reducing the learning rate.
            min_lr: Lower bound for the learning rate.
            verbose: Whether to log updates.
        """
        super().__init__(verbose)
        self.factor = factor
        self.nepochs = nepochs
        self.min_lr = min_lr

    def _show_props(self) -> dict[str, str]:
        props = super()._show_props()

        props["factor"] = str(self.factor)
        props["min-lr"] = str(self.min_lr)

        return props

    def on_epoch_end(self, train_loss: ndarray, val_loss: ndarray) -> None:
        """
        Update the learning rate at the end of an epoch if the interval is reached.

        Args:
            train_loss: Training loss array.
            val_loss: Validation loss array.
        """
        self.epoch_count += 1
        if (
            self.epoch_count % self.nepochs == 0
            and self.model.optimizer.learning_rate * self.factor >= self.min_lr
        ):
            self.model.optimizer.learning_rate *= self.factor
            self.log(f"Setting learning rate to {self.model.optimizer.learning_rate:.8f}!")

    @classmethod
    def from_model(cls, model: Model) -> ReduceLREveryNEpochs:
        """
        Create a scheduler instance from a model configuration.

        Args:
            model: The model instance containing scheduler parameters.

        Returns:
            An instance of ReduceLREveryNEpochs.
        """
        return ReduceLREveryNEpochs(
            model.reduce_lr_every_nepochs_factor,
            model.reduce_lr_every_nepochs_nepochs,
            model.reduce_lr_every_nepochs_min_lr,
        )
