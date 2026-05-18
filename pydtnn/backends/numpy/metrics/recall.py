"""
Numpy backend implementation of the Recall metric.
"""

import logging
import math
from typing import TYPE_CHECKING

from pydtnn.backends.numpy.metrics.abstract.metric import MetricNumpy
from pydtnn.backends.numpy.metrics.binary_confusion_matrix import BinaryConfusionMatrixNumpy
from pydtnn.libs import numpy as np
from pydtnn.metrics.recall import Recall

__all__ = ("RecallNumpy",)

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    import numpy as np

# from pydtnn.backends.numpy.utils.div_arrays_set_if_zero import div_arrays_set_if_zero


class RecallNumpy(Recall[np.ndarray], MetricNumpy):
    """
    Numpy implementation of the Recall metric for binary classification.
    """

    conf_matrix_metric: BinaryConfusionMatrixNumpy

    def _model_init(self) -> None:
        """
        Initializes model-specific parameters and calculates memory requirements.
        """
        super()._model_init()
        self.temp_var_shape = (self.shape[1],)
        self.tmp_memory_used += int(2 * math.prod(self.temp_var_shape)) * np.float32().itemsize
        self.tmp_memory_used += int(1 * math.prod(self.temp_var_shape)) * np.bool_().itemsize
        self.memory_used += self.tmp_memory_used

    def _post_init(self) -> None:
        """
        Allocates memory for internal buffers after model initialization.
        """
        super()._post_init()
        with self.model.memory:
            self.true_positives = self.model.memory.ndarray(self.temp_var_shape, dtype=np.float32)
            self.false_negatives = self.model.memory.ndarray(self.temp_var_shape, dtype=np.float32)
            self.are_zeros = self.model.memory.ndarray(self.temp_var_shape, dtype=np.bool_)

    def compute(self, y_pred: np.ndarray, y_targ: np.ndarray) -> float:
        """
        Computes the recall score based on predicted and target values.

        Args:
            y_pred: Predicted values.
            y_targ: Target ground truth values.

        Returns:
            The average recall score.
        """
        true_positives = self.true_positives
        false_negatives = self.false_negatives
        are_zeros = self.are_zeros
        # This two variables are not necessary, are to make the code more understandable.
        real_positives = false_negatives
        recall = false_negatives

        np.copyto(true_positives, self.conf_matrix_metric.get_true_positives())
        np.copyto(true_positives, self.conf_matrix_metric.get_false_negatives())
        # true_positives / (true_positives + false_negatives)
        np.add(true_positives, false_negatives, dtype=np.dtype(float), out=real_positives)
        # div_arrays_set_if_zero(recall,  divider, default_value=0.0)

        np.not_equal(real_positives, 0, out=are_zeros)
        np.divide(true_positives, real_positives, out=recall, where=(are_zeros))
        return np.average(recall).item()
