"""Module for schedulers that depend on model loss or metric values."""

import logging
import operator

from pydtnn.schedulers.abstract.scheduler import Scheduler

__all__ = ("SchedulerWithLossOrMetric",)

logger = logging.getLogger(__name__)


class SchedulerWithLossOrMetric(Scheduler):
    """Base class for schedulers that adjust based on specific loss or metric values."""

    def __init__(self, loss_or_metric: str, verbose: bool, minimize: bool) -> None:
        """
        Initializes the scheduler with a target metric and verbosity setting.

        Args:
            loss_or_metric (str): The name of the loss or metric to track.
            minimize (bool): Whether the metric should be minimized.
            verbose (bool): Whether to enable verbose logging.
        """
        # NOTE: loss_or_metric default value is "val_categorical_accuracy" in Parser.
        super().__init__(verbose)
        type, metric = loss_or_metric.split("_", 1)
        self.is_val_metric: bool = "val" == type
        self.loss_or_metric = metric
        self.minimize = minimize
        self.compare = operator.lt if self.minimize else operator.gt

    def _show_props(self) -> dict[str, str]:
        props = super()._show_props()

        type = "val" if self.is_val_metric else "train"
        props["metric"] = f"{type}_{self.loss_or_metric}"
        props["minimize"] = str(self.minimize)

        return props

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
                f"(Found: {self.model.loss_and_metric_names})"
            ) from e
