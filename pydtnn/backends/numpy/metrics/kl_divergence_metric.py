"""
Kullback-Leibler divergence metric implementation for the NumPy backend.
"""

import logging
import math
from typing import TYPE_CHECKING

from pydtnn.backends.numpy.metrics.metric import MetricNumpy
from pydtnn.libs import numpy as np
from pydtnn.metrics.kl_divergence_metric import KLDivergenceMetric

__all__ = ("KLDivergenceMetricNumpy",)

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    import numpy as np


class KLDivergenceMetricNumpy(KLDivergenceMetric[np.ndarray], MetricNumpy):
    """
    NumPy implementation of the KL Divergence metric.
    """

    def _model_init(self) -> None:
        """
        Initializes model-specific memory requirements for the metric.
        """
        super()._model_init()

        self.tmp_memory_used += int(math.prod(self.shape)) * self.model.dtype.itemsize
        self.memory_used += self.tmp_memory_used

    def _post_init(self) -> None:
        """
        Allocates memory for the loss buffer after model initialization.
        """
        super()._post_init()
        with self.model.memory:
            self.loss = self.model.memory.ndarray(self.shape, dtype=self.model.dtype)

    def compute(self, y_pred: np.ndarray, y_targ: np.ndarray) -> float:
        """
        Computes the KL divergence between predicted and target distributions.

        Args:
            y_pred: Predicted probability distribution.
            y_targ: Target probability distribution.

        Returns:
            The calculated KL divergence as a float.
        """
        y_targ = np.asarray(y_targ, dtype=self.model.dtype, order="C")
        loss = self.loss[: y_pred.shape[0]]
        # loss = np.abs(y_pred * np.log(np.abs(y_pred / (y_targ + eps) + eps)))
        np.add(y_targ, self.eps, out=loss)
        np.divide(y_pred, loss, out=loss)
        np.add(loss, self.eps, out=loss)
        np.abs(loss, out=loss)
        np.log(loss, out=loss)
        np.multiply(y_pred, loss, out=loss)
        np.abs(loss, out=loss)

        loss = np.sum(loss) / y_pred.shape[0]
        return float(loss)
