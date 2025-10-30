from typing import TYPE_CHECKING

import numpy as np

from pydtnn.schedulers.scheduler_with_loss_or_metric import SchedulerWithLossOrMetric

if TYPE_CHECKING:
    from pydtnn.model import Model


class ReduceLROnPlateau(SchedulerWithLossOrMetric):
    """
    ReduceLROnPlateau LRScheduler
    """

    def __init__(self, loss_or_metric: str = "", factor=0.1, patience=5, min_lr=0, verbose=True):
        # NOTE: loss_or_metric default value is "val_accuracy" in Parser.
        super().__init__(loss_or_metric, verbose)
        self.factor = factor
        self.patience = patience
        self.min_lr = min_lr
        self.best_epoch: int = 0
        self.best_loss: float = np.inf * {True: -1, False: 1}["accuracy" in self.loss_or_metric]

    def on_epoch_end(self, train_loss: np.ndarray[float], val_loss: np.ndarray[float]) -> None:
        idx = self._get_idx()
        self.epoch_count += 1
        loss = val_loss if self.is_val_metric else train_loss
        if ("accuracy" in self.loss_or_metric and loss[idx] > self.best_loss) or \
                ("accuracy" not in self.loss_or_metric and loss[idx] < self.best_loss):
            self.best_loss = loss[idx]
            self.best_epoch = self.epoch_count
        elif self.epoch_count - self.best_epoch >= self.patience \
                and self.model.optimizer.learning_rate * self.factor >= self.min_lr:
            self.model.optimizer.learning_rate *= self.factor
            self.best_epoch = self.epoch_count
            self.log("Metric '{}' did not improve for {} epochs, setting learning rate to {:.8f}."
                     .format(self.loss_or_metric, self.patience, self.model.optimizer.learning_rate))

    @classmethod
    def from_model(cls, model: "Model") -> "ReduceLROnPlateau":
        return ReduceLROnPlateau(model.reduce_lr_on_plateau_metric,
                                 model.reduce_lr_on_plateau_factor,
                                 model.reduce_lr_on_plateau_patience,
                                 model.reduce_lr_on_plateau_min_lr)
