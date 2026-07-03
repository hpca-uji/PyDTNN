"""Numpy backend implementation for Mean Absolute Error (MAE) regression metric."""

import logging
import math
from typing import TYPE_CHECKING

from pydtnn.backends.numpy.metrics.abstract.metric import MetricNumpy
from pydtnn.libs import numpy as np
from pydtnn.metrics.regression_mae import RegressionMAE

__all__ = ("RegressionMAENumpy",)

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    import numpy as np


class RegressionMAENumpy(RegressionMAE[np.ndarray], MetricNumpy):
    """Numpy-based implementation of the Mean Absolute Error metric."""

    def _model_init(self) -> None:
        """Initializes model memory requirements for the MAE calculation."""
        super()._model_init()

        self.tmp_memory_used += int(math.prod(self.shape)) * self.model.dtype.itemsize
        self.memory_used += self.tmp_memory_used

    def _post_init(self) -> None:
        """Allocates temporary memory buffers for difference calculations."""
        super()._post_init()
        with self.model.memory:
            self.diff = self.model.memory.ndarray(self.shape, dtype=self.model.dtype)

    def compute(self, y_pred: np.ndarray, y_targ: np.ndarray) -> float:
        """
        Computes the Mean Absolute Error between predictions and targets.

        Args:
            y_pred: Predicted values.
            y_targ: Ground truth target values.

        Returns:
            The calculated MAE as a float.
        """
        y_targ = np.asarray(y_targ, dtype=self.model.dtype, order="C")
        diff = self.diff[: y_pred.shape[0]]
        # return np.sum(np.absolute(y_targ - y_pred))
        np.subtract(y_targ, y_pred, dtype=self.model.dtype, out=diff)
        np.absolute(diff, out=diff, dtype=self.model.dtype, casting="unsafe")
        return np.mean(diff, dtype=self.model.dtype).item()
