"""
Module for managing model checkpointing during training.

This module provides the `ModelCheckpoint` class, a scheduler designed to
automatically save the state of a machine learning model at regular intervals
or when a monitored performance metric (like loss or accuracy) improves.
"""

from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING

import numpy as np

from pydtnn import timestamp
from pydtnn.schedulers.abstract.scheduler_with_loss_or_metric import SchedulerWithLossOrMetric

__all__ = ("ModelCheckpoint",)

logger = logging.getLogger(__name__)


if TYPE_CHECKING:
    from pydtnn.model import Model


class ModelCheckpoint(SchedulerWithLossOrMetric):
    """
    Scheduler that saves model states based on monitored loss or metric performance.

    This class extends `SchedulerWithLossOrMetric` to provide functionality for
    saving model checkpoints. It can save the model periodically based on
    `epoch_save_frequency` or when the monitored metric indicates an improvement.
    It also manages the deletion of older checkpoints to save disk space.
    """

    model: Model

    def __init__(
        self, loss_or_metric: str = "", epoch_save_frequency: int = 1, verbose: bool = True
    ) -> None:
        """
        Initializes the ModelCheckpoint scheduler.

        Args:
            loss_or_metric: The name of the loss or metric to monitor for
                            performance improvement. If empty, it defaults to
                            monitoring validation loss.
            epoch_save_frequency: The number of epochs between saving checkpoints,
                                  regardless of performance improvement.
            verbose: If True, logs checkpointing actions to the console/logger.
        """
        super().__init__(loss_or_metric, verbose)
        self.epoch_count = self.best_epoch = 0
        # Initialize best_loss to positive infinity for minimization metrics
        # and negative infinity for maximization metrics (like accuracy).
        self.best_loss = np.inf * {True: -1, False: 1}["accuracy" in self.loss_or_metric]
        # Attributes that will be properly defined elsewhere
        self.filename: str | None = None
        self.last_filename: str | None = None

    def on_epoch_end(self, train_loss: np.ndarray, val_loss: np.ndarray) -> None:
        """
        Evaluates performance at the end of an epoch and saves the model if the metric improves.

        This method is called automatically by the training loop after each epoch.
        It checks if the current epoch's performance metric is better than the
        best recorded performance. If it is, it updates the best performance
        and saves the model state. It also saves the model if the current epoch
        number is a multiple of `epoch_save_frequency`.

        Args:
            train_loss: A NumPy array containing the training loss values for each batch
                        in the current epoch.
            val_loss: A NumPy array containing the validation loss values for each batch
                      in the current epoch.
        """
        idx = self._get_idx()  # Determine the index for the monitored metric
        self.epoch_count += 1
        # Select the appropriate loss array based on whether we are monitoring
        # validation or training metrics.
        loss = val_loss if self.is_val_metric else train_loss

        # Check if the current performance is better than the best recorded.
        if self.compare(loss[idx], self.best_loss):
            self.best_loss = loss[idx]
            self.best_epoch = self.epoch_count
            # Save the model if the epoch count is a multiple of the save frequency.
            if (self.epoch_count % self.epoch_save_frequency) == 0:
                # Construct a unique filename including model name, epoch, and timestamp.
                self.filename = (
                    f"./model-{self.model.model_name}-epoch-{self.epoch_count}-{timestamp}.npz"
                )
                # Save the model's state.
                self.model.save_model_state(self.filename)
                self.log(f"Saving model weights and bias in '{self.filename}'.")
                # If this is a distributed training setup and we have a previous file,
                # remove the old checkpoint to save disk space.
                if self.model.comm_rank == 0 and self.last_filename is not None:
                    os.remove(self.last_filename)
                # Update last_filename to the newly saved checkpoint.
                self.last_filename = self.filename

    @classmethod
    def from_model(cls, model: Model) -> ModelCheckpoint:
        """
        Factory method to create a ModelCheckpoint instance from a model's configuration.

        This method simplifies the creation of a `ModelCheckpoint` scheduler by
        retrieving the necessary configuration parameters (metric to monitor and
        save frequency) directly from the provided `Model` instance.

        Args:
            model: The model instance from which to derive checkpointing configuration.
                   It is expected to have attributes like `model_checkpoint_metric`
                   and `model_checkpoint_save_freq`.

        Returns:
            A configured `ModelCheckpoint` instance ready to be used in training.
        """
        return ModelCheckpoint(model.model_checkpoint_metric, model.model_checkpoint_save_freq)
