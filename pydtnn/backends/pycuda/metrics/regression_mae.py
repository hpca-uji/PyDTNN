"""PyCUDA implementation of the Mean Absolute Error (MAE) regression metric."""

import logging

import numpy as np

from pydtnn.backends.pycuda.metrics.abstract.metric import MetricPycuda
from pydtnn.metrics.regression_mae import RegressionMAE
from pydtnn.utils.tensor_array import DescriptorType, MemoryType, TensorArray

__all__ = ("RegressionMAEPycuda",)

logger = logging.getLogger(__name__)


class RegressionMAEPycuda(RegressionMAE[TensorArray], MetricPycuda):
    """PyCUDA-accelerated Mean Absolute Error metric for regression tasks."""

    def _model_init(self) -> None:
        """Initializes the metric buffers and internal state for PyCUDA execution."""
        super()._model_init()

        n = self.model.batch_size
        num_classes = self.model.output_shape

        self.recall = TensorArray.new_zeros(
            (1,),
            self.model.dtype,
            tensor_format=self.model.tensor_format,
            mem_type=MemoryType.GLOBAL if self.model.use_gpudirect else MemoryType.DEVICE,
            desc_type=DescriptorType.TENSOR
        )
        self.memory_used += self.recall.nbytes
        self.local_recall = TensorArray.new_zeros(
            (n, *num_classes),
            self.model.dtype,
            tensor_format=self.model.tensor_format,
            mem_type=MemoryType.GLOBAL if self.model.use_gpudirect else MemoryType.DEVICE,
            desc_type=DescriptorType.TENSOR
        )
        self.memory_used += self.local_recall.nbytes

    def compute(self, y_pred: TensorArray, y_targ: TensorArray) -> float:
        """Computes the Mean Absolute Error between predictions and targets.

        Args:
            y_pred: Predicted values.
            y_targ: Ground truth target values.

        Returns:
            The calculated MAE as a float.
        """

        n = self.model.real_batch_size
        num_classes = y_pred.shape[1]

        self.res.fill(0)
        self.local_res.fill(0)

        n = np.int32(n)
        num_classes = np.int32(num_classes)
        self.kernel(
            y_targ.ary,
            y_pred.ary,
            self.res.ary,
            self.local_res.ary,
            n,
            num_classes,
            grid=self.grid,
            block=self.block,
            stream=self.model.pycuda_stream,
        )
        return self.res.get().item()
