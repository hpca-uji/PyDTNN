"""PyCUDA implementation of the Precision metric for PyDTNN."""

import logging

import numpy as np

from pydtnn.backends.pycuda.metrics.abstract.metric import MetricPycuda
from pydtnn.metrics.precision import Precision
from pydtnn.utils.tensor_array import FormatType, MemoryType, TensorArray

__all__ = ("PrecisionPycuda",)

logger = logging.getLogger(__name__)


class PrecisionPycuda(Precision[TensorArray], MetricPycuda):
    """Precision metric implementation using PyCUDA for GPU acceleration."""

    def _model_init(self) -> None:
        """Initializes the precision and local precision buffers on the GPU."""
        super()._model_init()
        target_classes = self.model.output_shape[0]
        self.precision = TensorArray.zeros(
            (1,),
            self.model.dtype,
            tensor_format=self.model.tensor_format,
            mem_type=MemoryType.GLOBAL if self.model.use_gpudirect else MemoryType.DEVICE,
            fmt_type=FormatType.TENSOR
        )
        self.memory_used += self.precision.nbytes
        self.local_precision = TensorArray.zeros(
            (target_classes,),
            self.model.dtype,
            tensor_format=self.model.tensor_format,
            mem_type=MemoryType.GLOBAL if self.model.use_gpudirect else MemoryType.DEVICE,
            fmt_type=FormatType.TENSOR
        )
        self.memory_used += self.local_precision.nbytes

    def compute(self, y_pred: TensorArray, y_targ: TensorArray) -> float:
        """
        Computes the precision score using a PyCUDA kernel.

        Args:
            y_pred: Predicted tensor array.
            y_targ: Target tensor array.

        Returns:
            The computed precision value as a float.
        """

        target_classes = self.model.output_shape[0]

        target_classes = np.int32(target_classes)

        self.precision.fill(0)
        self.local_precision.fill(0)

        self.kernel(
            self.precision.ary,
            self.conf_matrix_metric.conf_matrix.ary,
            self.local_precision.ary,
            target_classes,
            grid=self.grid,
            block=self.block,
            stream=self.model.pycuda_stream,
        )

        return self.precision.get().item()
