"""CuPy implementation of the Precision metric."""

import logging
from typing import TYPE_CHECKING

from pydtnn.backends.cupy.metrics.abstract.metric import MetricCupy
from pydtnn.backends.numpy.metrics.precision import PrecisionNumpy
from pydtnn.libs import numpy as np

__all__ = ("PrecisionCupy",)

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    import numpy as np


class PrecisionCupy(PrecisionNumpy, MetricCupy):
    """Precision metric implementation using CuPy backend."""

    def _post_init(self) -> None:
        """Initializes metric buffers on the GPU device."""
        super()._post_init()
        with self.model.memory:
            self.true_positives = self.model.memory.ndarray(self.temp_var_shape, dtype=self.dtype)
            self.false_positives = self.model.memory.ndarray(self.temp_var_shape, dtype=self.dtype)
            self.false_negatives = self.model.memory.ndarray(self.temp_var_shape, dtype=self.dtype)

    def compute(self, y_pred: np.ndarray, y_targ: np.ndarray) -> float:
        """Computes the precision score using GPU-accelerated operations.

        Args:
            y_pred: Predicted labels.
            y_targ: Ground truth labels.

        Returns:
            The average precision score.
        """
        true_positives = self.true_positives
        false_positives = self.false_positives
        # This two variables are not necessary, are to make the code more understandable.
        positives = false_positives
        precision = false_positives

        np.copyto(true_positives, self.conf_matrix_metric.get_true_positives())
        np.copyto(false_positives, self.conf_matrix_metric.get_false_positives())
        # true_positives / (true_positives + false_positives)

        np.add(true_positives, false_positives, out=positives)
        # precision = (precision / divider if divider[i] != 0 else default_value)
        # div_arrays_set_if_zero(precision,  f_positives, default_value=0)
        for i in range(true_positives.shape[0]):
            precision[i] = (true_positives[i] / positives[i]) if positives[i] != 0 else 0
        return np.average(precision).item()
