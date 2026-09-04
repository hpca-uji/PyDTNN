"""Warm-up learning rate scheduler module for PyDTNN."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from numpy import ndarray

from pydtnn.schedulers.abstract.scheduler import Scheduler

"""Warm-up learning rate scheduler module for PyDTNN."""

__all__ = ("WarmUp",)

logger = logging.getLogger(__name__)


if TYPE_CHECKING:
    from pydtnn.model import Model


class WarmUp(Scheduler):
    """Learning rate scheduler that linearly increases the learning rate during an initial warm-up phase."""

    def __init__(
        self,
        warmup_epochs: int = 5,
        base_lr: float = 1e-4,
        init_lr: float = 1e-3,
        verbose: bool = False,
    ) -> None:
        """
        Initialize the WarmUp scheduler.

        Args:
            warmup_epochs: Number of epochs to perform warm-up.
            base_lr: The starting learning rate.
            init_lr: The target learning rate after warm-up.
            verbose: Whether to log learning rate updates.
        """
        super().__init__(verbose)
        self.warmup_epochs = warmup_epochs
        self.base_lr = base_lr
        self.init_lr = init_lr
        self.epoch_count: int = 0

    def _show_props(self) -> dict[str, str]:
        props = super()._show_props()

        props["epochs"] = str(self.warmup_epochs)
        props["base-lr"] = str(self.base_lr)
        props["init-lr"] = str(self.init_lr)

        return props

    def on_epoch_end(self, train_loss: ndarray, val_loss: ndarray) -> None:
        """
        Update the model's learning rate at the end of each epoch if within the warm-up period.

        Args:
            train_loss: Training loss value.
            val_loss: Validation loss value.
        """
        if self.epoch_count < self.warmup_epochs:
            self.model.optimizer.learning_rate = self.base_lr + (
                (self.epoch_count + 1) / self.warmup_epochs
            ) * (self.init_lr - self.base_lr)
            self.epoch_count += 1
            self.log(f"Setting learning rate to {self.model.optimizer.learning_rate:.8f}.")

    @classmethod
    def from_model(cls, model: Model) -> WarmUp:
        """
        Create a WarmUp instance from a model configuration.

        Args:
            model: The model instance containing warm-up parameters.

        Returns:
            A configured WarmUp scheduler instance.
        """
        return WarmUp(model.warm_up_epochs, model.learning_rate / model.nprocs, model.learning_rate)
