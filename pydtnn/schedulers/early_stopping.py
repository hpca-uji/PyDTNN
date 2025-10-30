from typing import TYPE_CHECKING
import time

import numpy as np

from pydtnn.schedulers.scheduler_with_loss_or_metric import SchedulerWithLossOrMetric

if TYPE_CHECKING:
    from pydtnn.model import Model


class EarlyStopping(SchedulerWithLossOrMetric):
    """
    EarlyStopping LRScheduler
    """

    def __init__(self, loss_or_metric="", patience=10, minimize=True, verbose=True):
        super().__init__(loss_or_metric, verbose)
        self.patience = patience
        self.minimize = minimize
        self.best_epoch: int = 0
        self.best_loss: float = np.inf * {True: -1, False: 1}[not self.minimize]
        self.best_weights_filename: str | None = None
        self.time: str = time.strftime("%Y%m%d")

    def on_epoch_end(self, train_loss: np.ndarray[float], val_loss: np.ndarray[float]) -> None:
        idx = self._get_idx()
        self.epoch_count += 1
        loss = val_loss if self.is_val_metric else train_loss
        if (not self.minimize and loss[idx] > self.best_loss) or \
                (self.minimize and loss[idx] < self.best_loss):
            self.best_loss = loss[idx]
            self.best_epoch = self.epoch_count
            # Save weights + bias
            if not self.best_weights_filename:
                self.best_weights_filename = "./model-{}-weights-rank_{}-{}.npz" \
                    .format(self.model.model_name, self.model.comm_rank, self.time)
            self.model.store_weights_and_bias(self.best_weights_filename, compress=False)
        elif (self.epoch_count - self.best_epoch) >= self.patience:
            self.stop_training = True
            # Restore weights + bias
            self.model.load_weights_and_bias(self.best_weights_filename)
            self.log(f"Metric '{self.loss_or_metric}' did not improve for {self.patience} epochs, stop training.")

    @classmethod
    def from_model(cls, model: "Model") -> "EarlyStopping":
        return EarlyStopping(model.early_stopping_metric,
                             model.early_stopping_patience,
                             model.early_stopping_minimize)
