"""PyCUDA implementation of the Mean Absolute Error (MAE) regression metric."""

import logging

import numpy as np

from pydtnn.backends.pycuda.metrics.metric import MetricPycuda
from pydtnn.backends.pycuda.utils.tensor_array import TensorArray
from pydtnn.metrics.regression_mae import RegressionMAE

__all__ = ("RegressionMAEPycuda",)

logger = logging.getLogger(__name__)


class RegressionMAEPycuda(RegressionMAE[TensorArray], MetricPycuda):
    """PyCUDA-accelerated Mean Absolute Error metric for regression tasks."""

    def _model_init(self) -> None:
        """Initializes the metric buffers and internal state for PyCUDA execution."""
        super()._model_init()

        n = self.model.batch_size
        num_classes = self.model.output_shape

        self.res = TensorArray.new_zeros(shape=(1,), dtype=np.dtype(self.model.dtype), tensor_format=self.model.tensor_format, cudnn_dtype=self.model.cudnn_dtype)

        self.local_res = TensorArray.new_zeros(shape=(n, *num_classes), dtype=np.dtype(self.model.dtype), tensor_format=self.model.tensor_format, cudnn_dtype=self.model.cudnn_dtype)

    def compute(self, y_pred: TensorArray, y_targ: TensorArray) -> float:
        """Computes the Mean Absolute Error between predictions and targets.

        Args:
            y_pred: Predicted values.
            y_targ: Ground truth target values.

        Returns:
            The calculated MAE as a float.
        """

        n = y_pred.shape[0]
        num_classes = y_pred.shape[1]

        self.res.fill(0)
        self.local_res.fill(0)

        n = np.int32(n)
        num_classes = np.int32(num_classes)
        self.kernel(y_targ.ary, y_pred.ary, self.res.ary, self.local_res.ary, n, num_classes, grid=self.grid, block=self.block, stream=self.model.stream)
        return self.res.get().item()
