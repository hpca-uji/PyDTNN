"""Numpy backend implementation of the categorical accuracy metric."""

import logging
import math
from typing import TYPE_CHECKING

from pydtnn.backends.numpy.metrics.abstract.metric import MetricNumpy
from pydtnn.libs import numpy as np
from pydtnn.metrics.categorical_accuracy import CategoricalAccuracy

__all__ = ("CategoricalAccuracyNumpy",)

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    import numpy as np  # noqa: F811 (override typing)


class CategoricalAccuracyNumpy(CategoricalAccuracy[np.ndarray], MetricNumpy):
    """Numpy-based implementation of categorical accuracy."""

    def _model_init(self) -> None:
        """Initializes model-specific parameters and calculates memory usage."""
        super()._model_init()
        self._argmax_shape = (self.model.batch_size,)
        self.tmp_memory_used = int(math.prod(self._argmax_shape)) * np.int32().itemsize
        self.memory_used += self.tmp_memory_used  # + arange_size = self.model.batch_size

    def _post_init(self) -> None:
        """Allocates memory for the argmax buffer after model initialization."""
        super()._post_init()
        with self.model.memory:
            self._argmax = self.model.memory.ndarray(self._argmax_shape, dtype=np.int32)

    def compute(self, y_pred: np.ndarray, y_targ: np.ndarray) -> float:
        """
        Computes the categorical accuracy between predictions and targets.

        Args:
            y_pred: Predicted probabilities or logits.
            y_targ: Ground truth labels.

        Returns:
            The calculated accuracy as a percentage.
        """
        y_targ = np.asarray(y_targ, dtype=self.model.dtype, order="C")
        b = y_targ.shape[0]
        _argmax = self._argmax[:b]
        # return np.sum(y_targ[np.arange(b), np.argmax(y_pred, axis=1)]) * 100 / b

        np.argmax(y_pred, axis=1, out=_argmax)
        y: np.ndarray = y_targ[np.arange(b), _argmax]
        y = np.sum(y, dtype=self.model.dtype)
        y *= 100 / b
        return y.item()
