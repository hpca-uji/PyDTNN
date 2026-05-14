"""
Numpy implementation of the F1 Score metric for the PyDTNN framework.
"""

import logging
import math
from typing import TYPE_CHECKING

from pydtnn.backends.numpy.metrics.binary_confusion_matrix import BinaryConfusionMatrixNumpy
from pydtnn.backends.numpy.metrics.metric import MetricNumpy
from pydtnn.libs import numpy as np
from pydtnn.metrics.f1_score import F1Score

__all__ = ("F1ScoreNumpy",)


logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    import numpy as np


class F1ScoreNumpy(F1Score[np.ndarray], MetricNumpy):
    """
    Numpy-based F1 Score metric calculation.
    """

    conf_matrix_metric: BinaryConfusionMatrixNumpy

    def _model_init(self) -> None:
        """
        Initializes model memory requirements for F1 score calculation.
        """
        super()._model_init()
        shape = self.shape[1]

        self.temp_var_shape = (shape,)
        self.tmp_memory_used += int(3 * math.prod(self.temp_var_shape)) * np.float32().itemsize
        self.tmp_memory_used += int(1 * math.prod(self.temp_var_shape)) * np.bool_().itemsize
        self.memory_used += self.tmp_memory_used

    def _post_init(self) -> None:
        """
        Allocates memory for intermediate buffers used during computation.
        """
        super()._post_init()
        with self.model.memory:
            self.true_positives = self.model.memory.ndarray(self.temp_var_shape, dtype=np.float32)
            self.false_positives = self.model.memory.ndarray(self.temp_var_shape, dtype=np.float32)
            self.false_negatives = self.model.memory.ndarray(self.temp_var_shape, dtype=np.float32)
            self.are_zeros = self.model.memory.ndarray(self.temp_var_shape, dtype=np.bool_)

    def compute(self, y_pred: np.ndarray, y_targ: np.ndarray) -> float:
        """
        Computes the F1 score based on confusion matrix statistics.

        Args:
            y_pred: Predicted labels.
            y_targ: Target labels.

        Returns:
            The average F1 score across classes.
        """
        true_positives = self.true_positives
        false_positives = self.false_positives
        false_negatives = self.false_negatives
        are_zeros = self.are_zeros

        # This variable is not necessary, is to make the code more understandable.
        aggregation = false_positives
        f1 = aggregation

        true_positives[:] = self.conf_matrix_metric.get_true_positives()
        false_positives[:] = self.conf_matrix_metric.get_false_positives()
        false_negatives[:] = self.conf_matrix_metric.get_false_negatives()

        # f1 =  2 * true_positives / (2 * true_positives + false_positives + false_negatives
        np.multiply(2, true_positives, out=true_positives)
        np.add(true_positives, false_positives, out=aggregation)
        np.add(aggregation, false_negatives, out=aggregation)

        # div_arrays_set_if_zero(true_positives,  aggregation, default_value=0.0)
        np.not_equal(aggregation, 0, out=are_zeros)
        np.divide(true_positives, aggregation, out=f1, where=(are_zeros))

        return np.average(f1).item()
