"""
PyCUDA implementation of the Categorical Mean Absolute Error metric.
"""

import logging

import numpy as np

from pydtnn.backends.pycuda.metrics.metric import MetricPycuda
from pydtnn.backends.pycuda.utils.tensor_array import TensorArray
from pydtnn.metrics.categorical_mae import CategoricalMAE

__all__ = ("CategoricalMAEPycuda",)

logger = logging.getLogger(__name__)


class CategoricalMAEPycuda(CategoricalMAE[TensorArray], MetricPycuda):
    """
    Categorical Mean Absolute Error metric implemented for PyCUDA backends.
    """

    def _model_init(self) -> None:
        """
        Initializes the PyCUDA-specific buffers for metric calculation.
        """
        super()._model_init()
        self.res = TensorArray.new_zeros(shape=(1,), dtype=np.dtype(self.model.dtype), tensor_format=self.model.tensor_format, cudnn_dtype=self.model.cudnn_dtype)
        self.local_res = TensorArray.new_zeros(shape=(self.model.batch_size,), dtype=np.dtype(self.model.dtype), tensor_format=self.model.tensor_format, cudnn_dtype=self.model.cudnn_dtype)

    def compute(self, y_pred: TensorArray, y_targ: TensorArray) -> float:
        """
        Computes the categorical MAE between predictions and targets using a CUDA kernel.

        Args:
            y_pred: Predicted tensor array.
            y_targ: Target tensor array.

        Returns:
            The computed mean absolute error as a float.
        """
        n = y_pred.shape[0]

        self.res.fill(0)
        self.local_res.fill(0)

        n = np.int32(n)
        num_classes = np.int32(y_pred.shape[1])

        self.kernel(y_targ.ary, y_pred.ary, self.res.ary, self.local_res.ary, n, num_classes, grid=self.grid, block=self.block, stream=self.model.stream)
        return float(self.res.get())
