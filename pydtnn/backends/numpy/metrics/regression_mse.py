"""Numpy backend implementation for Mean Squared Error regression metric."""

import logging
import math
from typing import TYPE_CHECKING

from pydtnn.backends.numpy.metrics.abstract.metric import MetricNumpy
from pydtnn.libs import numpy as np
from pydtnn.metrics.regression_mse import RegressionMSE

__all__ = ("RegressionMSENumpy",)

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    import numpy as np


class RegressionMSENumpy(RegressionMSE[np.ndarray], MetricNumpy):
    """Numpy-based implementation of the Mean Squared Error metric."""

    def _model_init(self) -> None:
        """Initializes model-specific memory requirements for the metric."""
        super()._model_init()

        self.tmp_memory_used += int(math.prod(self.shape)) * self.model.dtype.itemsize
        self.memory_used += self.tmp_memory_used

    def _post_init(self) -> None:
        """Allocates memory buffers for difference calculations after model initialization."""
        super()._post_init()
        with self.model.memory:
            self.diff = self.model.memory.ndarray(self.shape, dtype=self.model.dtype)

    def compute(self, y_pred: np.ndarray, y_targ: np.ndarray) -> float:
        """
        Computes the Mean Squared Error between predicted and target values.

        Args:
            y_pred: Predicted values.
            y_targ: Target ground truth values.

        Returns:
            The calculated MSE as a float.
        """
        y_targ = np.asarray(y_targ, dtype=self.model.dtype, order="C")
        diff: np.ndarray = self.diff[: y_pred.shape[0]]
        # return np.square(y_targ - y_pred).mean()
        np.subtract(y_targ, y_pred, dtype=self.model.dtype, out=diff)
        np.square(diff, out=diff, dtype=self.model.dtype, casting="unsafe")
        return diff.mean(dtype=self.model.dtype).item()
