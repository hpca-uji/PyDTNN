"""
Recall metric implementation for binary classification tasks.
"""

import logging

from pydtnn.metrics.abstract.metric import Metric
from pydtnn.metrics.binary_confusion_matrix import BinaryConfusionMatrix
from pydtnn.utils.constants import Array

__all__ = ("Recall",)

logger = logging.getLogger(__name__)


class Recall[T: Array](Metric[T]):
    """
    Calculates the recall score based on a binary confusion matrix.
    """

    order = BinaryConfusionMatrix.order + 1
    conf_matrix_metric: BinaryConfusionMatrix = None  # type: ignore
    format = "rec: %.4f"

    def _model_init(self) -> None:
        """
        Initializes the metric by locating the required BinaryConfusionMatrix in the model.
        """

        for metric in self.model.metrics_funcs:
            if isinstance(metric, BinaryConfusionMatrix):
                self.conf_matrix_metric = metric
                break
        assert self.conf_matrix_metric is not None, (
            f"Recall requires of {BinaryConfusionMatrix.__name__}"
        )
        super()._model_init()
