"""Module for schedulers that depend on model loss or metric values."""

import logging
import operator

from pydtnn.schedulers.abstract.scheduler import Scheduler

__all__ = ("SchedulerWithLossOrMetric",)

logger = logging.getLogger(__name__)


class SchedulerWithLossOrMetric(Scheduler):
    """Base class for schedulers that adjust based on specific loss or metric values."""

    def __init__(self, loss_or_metric: str, verbose: bool) -> None:
        """
        Initializes the scheduler with a target metric and verbosity setting.

        Args:
            loss_or_metric: The name of the loss or metric to track.
            verbose: Whether to enable verbose logging.
        """
        # NOTE: loss_or_metric default value is "val_accuracy" in Parser.
        super().__init__(verbose)
        type, metric = loss_or_metric.split("_", 1)
        self.is_val_metric: bool = "val" == type
        self.loss_or_metric = metric
        self.compare = operator.lt if "accuracy" in self.loss_or_metric else operator.gt

    def _get_idx(self) -> int:
        """
        Retrieves the index of the tracked metric within the model's metrics list.

        Returns:
            The index of the metric.

        Raises:
            ValueError: If the metric is not found in the model.
        """
        try:
            return self.model.loss_and_metric_names.index(self.loss_or_metric)
        except ValueError as e:
            raise ValueError(
                f"{self}: loss or metric '{self.loss_or_metric}' not found in current model!"
            ) from e
