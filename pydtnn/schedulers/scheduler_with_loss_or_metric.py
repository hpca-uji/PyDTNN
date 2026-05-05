import logging
from typing import TYPE_CHECKING

from pydtnn.schedulers.scheduler import Scheduler

__all__ = (
    "SchedulerWithLossOrMetric",
)

logger = logging.getLogger(__name__)


if TYPE_CHECKING:
    from pydtnn.model import Model


class SchedulerWithLossOrMetric(Scheduler):
    """
    Scheduler with metric base class
    """

    def __init__(self, loss_or_metric: str, verbose: bool):
        # NOTE: loss_or_metric default value is "val_accuracy" in Parser.
        super().__init__(verbose)
        type, metric = loss_or_metric.split("_", 1)
        self.is_val_metric: bool = "val" == type
        self.loss_or_metric = metric

    def _get_idx(self):
        try:
            return self.model.loss_and_metrics.index(self.loss_or_metric)
        except ValueError as e:
            raise ValueError(f"{self}: loss or metric '{self.loss_or_metric}' not found in current model!") from e
