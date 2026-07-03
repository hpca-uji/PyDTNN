"""CuPy implementation of the Recall metric."""

import logging
from typing import TYPE_CHECKING

from pydtnn.backends.cupy.metrics.abstract.metric import MetricCupy
from pydtnn.backends.numpy.metrics.recall import RecallNumpy
from pydtnn.libs import numpy as np

__all__ = ("RecallCupy",)

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    import numpy as np  # noqa: F811 (override typing)


class RecallCupy(RecallNumpy, MetricCupy):
    """Recall metric implementation using CuPy backend."""

    def _post_init(self) -> None:
        """Initialize metric buffers on the device."""
        super()._post_init()
        with self.model.memory:
            self.true_positives = self.model.memory.ndarray(self.temp_var_shape, dtype=self.dtype)
            self.false_positives = self.model.memory.ndarray(self.temp_var_shape, dtype=self.dtype)
            self.false_negatives = self.model.memory.ndarray(self.temp_var_shape, dtype=self.dtype)

    def compute(self, y_pred: np.ndarray, y_targ: np.ndarray) -> float:
        """Compute the recall score based on predicted and target arrays.

        Args:
            y_pred: Predicted labels.
            y_targ: Ground truth labels.

        Returns:
            The average recall score.
        """
        true_positives = self.true_positives
        false_negatives = self.false_negatives
        # This two variables are not necessary, are to make the code more understandable.
        real_positives = false_negatives
        recall = false_negatives

        np.copyto(true_positives, self.conf_matrix_metric.get_true_positives())
        np.copyto(true_positives, self.conf_matrix_metric.get_false_negatives())
        # true_positives / (true_positives + false_negatives)
        np.add(true_positives, false_negatives, dtype=np.dtype(float), out=real_positives)
        # div_arrays_set_if_zero(recall,  divider, default_value=0.0)

        for i in range(true_positives.shape[0]):
            recall[i] = (true_positives[i] / real_positives[i]) if real_positives[i] != 0 else 0
        return np.average(recall).item()
