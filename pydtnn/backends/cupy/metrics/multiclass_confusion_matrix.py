"""
CuPy implementation of the multiclass confusion matrix metric.
"""
import logging

import numpy as np

from pydtnn.backends.cupy.metrics.metric import MetricCupy
from pydtnn.backends.numpy.metrics.multiclass_confusion_matrix import MulticlassConfusionMatrixNumpy

__all__ = ("MulticlassConfusionMatrixCupy",)


logger = logging.getLogger(__name__)


class MulticlassConfusionMatrixCupy(MulticlassConfusionMatrixNumpy, MetricCupy):
    """
    Computes the multiclass confusion matrix using CuPy-compatible arrays.
    """
    def compute(self, y_pred: np.ndarray, y_targ: np.ndarray) -> np.ndarray:
        """
        Computes the confusion matrix for multiclass classification.

        Args:
            y_pred: Predicted labels.
            y_targ: Ground truth labels.

        Returns:
            The computed confusion matrix as a CuPy array.
        """
        return np.asarray(super().compute(y_pred, y_targ))