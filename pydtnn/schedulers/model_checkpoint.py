from typing import TYPE_CHECKING
import os
import time

import numpy as np

from pydtnn.schedulers.scheduler_with_loss_or_metric import SchedulerWithLossOrMetric

if TYPE_CHECKING:
    from pydtnn.model import Model


class ModelCheckpoint(SchedulerWithLossOrMetric):
    """
    ModelCheckpoint LRScheduler
    """

    def __init__(self, loss_or_metric: str = "", epoch_save_frequency=1, verbose=True):
        super().__init__(loss_or_metric, verbose)
        self.epoch_save_frequency = epoch_save_frequency
        self.epoch_count = self.best_epoch = 0
        self.best_loss = np.inf * {True: -1, False: 1}["accuracy" in self.loss_or_metric]
        # Attributes that will be properly defined elsewhere
        self.filename: str | None = None
        self.last_filename: str | None = None

    def on_epoch_end(self, train_loss: np.ndarray[float], val_loss: np.ndarray[float]) -> None:
        idx = self._get_idx()
        self.epoch_count += 1
        loss = val_loss if self.is_val_metric else train_loss
        if ("accuracy" in self.loss_or_metric and loss[idx] > self.best_loss) or \
                ("accuracy" not in self.loss_or_metric and loss[idx] < self.best_loss):
            self.best_loss = loss[idx]
            self.best_epoch = self.epoch_count
            if self.epoch_count % self.epoch_save_frequency == 0:
                self.filename = "./model-{}-epoch-{}-{}.npz" \
                    .format(self.model.model_name, self.epoch_count, time.strftime("%Y%m%d"))
                self.model.store_weights_and_bias(self.filename)
                self.log(f"Saving model weights and bias in '{self.filename}'.")
                if self.model.comm_rank == 0 and self.last_filename is not None:
                    os.remove(self.last_filename)
                self.last_filename = self.filename

    @classmethod
    def from_model(cls, model: "Model") -> "ModelCheckpoint":
        return ModelCheckpoint(model.model_checkpoint_metric,
                               model.model_checkpoint_save_freq)
