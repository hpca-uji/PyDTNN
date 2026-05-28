"""
PyCUDA implementation of categorical accuracy metric.
"""

import logging

import numpy as np
from pycuda import gpuarray  # type: ignore

from pydtnn.backends.pycuda.metrics.abstract.metric import MetricPycuda
from pydtnn.backends.pycuda.utils.tensor_array import TensorArray
from pydtnn.metrics.categorical_accuracy import CategoricalAccuracy

__all__ = ("CategoricalAccuracyPycuda",)

logger = logging.getLogger(__name__)


class CategoricalAccuracyPycuda(CategoricalAccuracy[TensorArray], MetricPycuda):
    """
    Categorical accuracy metric implemented for PyCUDA backends.
    """

    def _model_init(self) -> None:
        """
        Initializes the metric buffers on the GPU.
        """
        super()._model_init()
        self.cost = gpuarray.zeros((self.model.batch_size,), self.model.dtype)

    def compute(self, y_pred: TensorArray, y_targ: TensorArray) -> float:
        """
        Computes the categorical accuracy between predictions and targets.

        Args:
            y_pred: Predicted tensor array.
            y_targ: Target tensor array.

        Returns:
            The calculated accuracy as a percentage.
        """
        self.kernel(
            y_targ.ary,
            y_pred.ary,
            self.cost,
            np.int32(self.model.batch_size),
            np.int32(self.shape[1]),
            grid=self.grid,
            block=self.block,
            stream=self.model.stream,
        )
        return float(gpuarray.sum(self.cost).get() * 100 / self.model.batch_size)
