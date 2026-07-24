"""Numpy backend implementation of the Categorical Mean Absolute Error metric."""

import logging
import math
from typing import TYPE_CHECKING

from pydtnn.backends.numpy.metrics.abstract.metric import MetricNumpy
from pydtnn.libs import numpy as np
from pydtnn.metrics.categorical_mae import CategoricalMAE

__all__ = ("CategoricalMAENumpy",)

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    import numpy as np  # noqa: F811 (override typing)


class CategoricalMAENumpy(CategoricalMAE[np.ndarray], MetricNumpy):
    """Numpy-based implementation of Categorical Mean Absolute Error."""

    def _model_init(self) -> None:
        """Initializes model parameters and calculates memory requirements."""
        super()._model_init()

        self.error: np.ndarray = None
        self.tmp_memory_used += int(math.prod(self.shape)) * self.model.dtype.itemsize
        self.memory_used += self.tmp_memory_used

    def _post_init(self) -> None:
        """Allocates memory for error storage after model initialization."""
        super()._post_init()
        with self.model.memory:
            self.error = self.model.memory.ndarray(self.shape, dtype=self.model.dtype)

    def compute(self, y_pred: np.ndarray, y_targ: np.ndarray) -> float:
        """
        Computes the categorical mean absolute error between predictions and targets.

        Args:
            y_pred: Predicted values.
            y_targ: Ground truth target values.

        Returns:
            The calculated mean absolute error as a float.
        """
        y_targ = np.asarray(y_targ, dtype=self.model.dtype, order="C")
        error = self.error[: y_pred.shape[0]]
        # return np.sum(np.absolute(1 - y_pred[np.arange(b), np.argmax(y_targ, axis=1)]))
        np.subtract(y_pred, y_targ, dtype=self.model.dtype, out=error)
        np.absolute(error, out=error, dtype=self.model.dtype)
        return np.mean(error, dtype=self.model.dtype).item()
