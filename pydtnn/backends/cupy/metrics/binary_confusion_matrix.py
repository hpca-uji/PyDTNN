"""CuPy implementation of the binary confusion matrix metric."""

import logging
from typing import TYPE_CHECKING

from pydtnn.libs import numpy as np
from pydtnn.backends.cupy.metrics.abstract.metric import MetricCupy
from pydtnn.backends.numpy.metrics.binary_confusion_matrix import BinaryConfusionMatrixNumpy

__all__ = ("BinaryConfusionMatrixCupy",)

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    import numpy as np  # noqa: F811 (override typing)


class BinaryConfusionMatrixCupy(BinaryConfusionMatrixNumpy, MetricCupy):
    """Computes the binary confusion matrix using CuPy-compatible operations."""

    def compute(self, y_pred: np.ndarray, y_targ: np.ndarray) -> np.ndarray:
        """
        Computes the confusion matrix for binary classification.

        Args:
            y_pred: Predicted labels.
            y_targ: Ground truth labels.

        Returns:
            A CuPy array containing the confusion matrix.
        """
        return np.asarray(super().compute(y_pred, y_targ))
