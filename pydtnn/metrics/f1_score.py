"""F1-Score metric implementation for binary classification tasks."""

import logging

from pydtnn.metrics.abstract.metric import Metric
from pydtnn.metrics.binary_confusion_matrix import BinaryConfusionMatrix
from pydtnn.utils.constants import Array

__all__ = ("F1Score",)

logger = logging.getLogger(__name__)


class F1Score[T: Array](Metric[T]):  # noqa: D101 (generics not detected)
    """Computes the F1-score based on a BinaryConfusionMatrix."""

    order = BinaryConfusionMatrix.order + 1
    conf_matrix_metric: BinaryConfusionMatrix = None  # pyright: ignore[reportAssignmentType]
    format = "f1: %.4f"

    def _model_init(self) -> None:
        """Initializes the metric by locating the required BinaryConfusionMatrix in the model."""

        for metric in self.model.metrics_funcs:
            if isinstance(metric, BinaryConfusionMatrix):
                self.conf_matrix_metric = metric
                break
        assert self.conf_matrix_metric is not None, (
            f"F1-Score requires of {BinaryConfusionMatrix.__name__}"
        )
        super()._model_init()
