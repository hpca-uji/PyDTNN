from typing import TYPE_CHECKING

from pydtnn.schedulers.scheduler import Scheduler

if TYPE_CHECKING:
    from pydtnn.model import Model


class SchedulerWithLossOrMetric(Scheduler):
    """
    Scheduler with metric base class
    """

    def __init__(self, loss_or_metric: str, verbose: bool):
        # NOTE: loss_or_metric default value is "val_accuracy" in Parser.
        super().__init__(verbose)
        self.is_val_metric: bool = "val_" == loss_or_metric[:4]
        self.loss_or_metric = loss_or_metric[4:] if self.is_val_metric else loss_or_metric

    def _get_idx(self):
        try:
            return self.model.loss_and_metrics.index(self.loss_or_metric)
        except ValueError as e:
            raise ValueError("{self}: loss or metric '{self.loss_or_metric}' not found in current model!") from e

    @classmethod
    def from_model(cls, model: "Model") -> "WarmUpLRScheduler":
        return 