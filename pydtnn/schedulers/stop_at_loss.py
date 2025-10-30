from typing import TYPE_CHECKING

from numpy import ndarray

from pydtnn.schedulers.scheduler_with_loss_or_metric import SchedulerWithLossOrMetric

if TYPE_CHECKING:
    from pydtnn.model import Model


class StopAtLoss(SchedulerWithLossOrMetric):
    """
    StopAtLoss LRScheduler
    """

    def __init__(self, loss_or_metric: str = "", threshold_value=0, verbose=True):
        # NOTE: loss_or_metric default value is "val_accuracy" in Parser.
        super().__init__(loss_or_metric, verbose)
        self.threshold_value = threshold_value

    def on_epoch_end(self, train_loss: ndarray[float], val_loss: ndarray[float]) -> None:
        idx = self._get_idx()
        self.epoch_count += 1
        loss = val_loss if self.is_val_metric else train_loss
        if ("accuracy" in self.loss_or_metric and loss[idx] > self.threshold_value) or \
                ("accuracy" not in self.loss_or_metric and loss[idx] < self.threshold_value):
            self.stop_training = True
            self.log("Metric '{self.loss_or_metric}' reached threshold value {self.threshold_value}, stop training.")

    @classmethod
    def from_model(cls, model: "Model") -> "StopAtLoss":
        return StopAtLoss(model.stop_at_loss_metric,
                          model.stop_at_loss_threshold)
