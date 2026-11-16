"""PyCUDA implementation of the Categorical Mean Absolute Error metric."""

import logging

import numpy as np

from pydtnn.backends.pycuda.metrics.abstract.metric import MetricPycuda
from pydtnn.metrics.categorical_mae import CategoricalMAE
from pydtnn.utils.tensor_array import DescriptorType, MemoryType, TensorArray

__all__ = ("CategoricalMAEPycuda",)

logger = logging.getLogger(__name__)


class CategoricalMAEPycuda(CategoricalMAE[TensorArray], MetricPycuda):
    """Categorical Mean Absolute Error metric implemented for PyCUDA backends."""

    def _model_init(self) -> None:
        """Initializes the PyCUDA-specific buffers for metric calculation."""
        super()._model_init()
        self.res = TensorArray.zeros(
            (1,),
            self.model.dtype,
            tensor_format=self.model.tensor_format,
            mem_type=MemoryType.GLOBAL if self.model.use_gpudirect else MemoryType.DEVICE,
            desc_type=DescriptorType.TENSOR
        )
        self.memory_used += self.res.nbytes
        self.local_res = TensorArray.zeros(
            (self.model.batch_size,),
            self.model.dtype,
            tensor_format=self.model.tensor_format,
            mem_type=MemoryType.GLOBAL if self.model.use_gpudirect else MemoryType.DEVICE,
            desc_type=DescriptorType.TENSOR
        )
        self.memory_used += self.local_res.nbytes

    def compute(self, y_pred: TensorArray, y_targ: TensorArray) -> float:
        """
        Computes the categorical MAE between predictions and targets using a CUDA kernel.

        Args:
            y_pred: Predicted tensor array.
            y_targ: Target tensor array.

        Returns:
            The computed mean absolute error as a float.
        """
        n = self.model.real_batch_size

        self.res.fill(0)
        self.local_res.fill(0)

        n = np.int32(n)
        num_classes = np.int32(y_pred.shape[1])

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
