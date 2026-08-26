"""Precision metric implementation for binary classification tasks."""

import logging

from pydtnn.metrics.abstract.metric import Metric
from pydtnn.metrics.binary_confusion_matrix import BinaryConfusionMatrix
from pydtnn.utils.constants import Array

__all__ = ("Precision",)

logger = logging.getLogger(__name__)


class Precision[T: Array](Metric[T]):  # noqa: D101 (generics not detected)
    """
    Calculates the precision metric based on a binary confusion matrix.

    Precision is defined as the ratio of true positives to the sum of true
    positives and false positives.
    """

    conf_matrix_metric: BinaryConfusionMatrix[T] = None  # pyright: ignore[reportAssignmentType]

    def order(self) -> int:
        for metric in self.model.metrics_funcs:
            if isinstance(metric, BinaryConfusionMatrix):
                break
        else:
            return 0
        return metric.order() + 1

    def _model_init(self) -> None:
        """Initializes the metric by locating the required BinaryConfusionMatrix within the model's metrics."""
        super()._model_init()
        for metric in self.model.metrics_funcs:
            if isinstance(metric, BinaryConfusionMatrix):
                self.conf_matrix_metric = metric
                break
        assert self.conf_matrix_metric is not None, (
            f"Precision requires of {BinaryConfusionMatrix.__name__}"
        )
