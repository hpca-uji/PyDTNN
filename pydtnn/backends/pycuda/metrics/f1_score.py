"""PyCUDA implementation of the F1-Score metric for the PyDTNN framework."""

import logging

import numpy as np

from pydtnn.backends.pycuda.metrics.abstract.metric import MetricPycuda
from pydtnn.metrics.f1_score import F1Score
from pydtnn.utils.tensor_array import FormatType, MemoryType, TensorArray

__all__ = ("F1ScorePycuda",)

logger = logging.getLogger(__name__)


class F1ScorePycuda(F1Score[TensorArray], MetricPycuda):
    """PyCUDA-accelerated F1-Score metric implementation."""

    def _model_init(self) -> None:
        """Initializes the F1-Score metric buffers on the GPU."""
        super()._model_init()
        target_classes = self.model.output_shape[0]
        self.f1 = TensorArray.zeros(
            (1,),
            self.model.dtype,
            tensor_format=self.model.tensor_format,
            mem_type=MemoryType.GLOBAL if self.model.use_gpudirect else MemoryType.DEVICE,
            fmt_type=FormatType.TENSOR
        )
        self.memory_used += self.f1.nbytes
        self.local_f1 = TensorArray.zeros(
            (target_classes,),
            self.model.dtype,
            tensor_format=self.model.tensor_format,
            mem_type=MemoryType.GLOBAL if self.model.use_gpudirect else MemoryType.DEVICE,
            fmt_type=FormatType.TENSOR
        )
        self.memory_used += self.local_f1.nbytes

    def compute(self, y_pred: TensorArray, y_targ: TensorArray) -> float:
        """
        Computes the F1-Score using a PyCUDA kernel.

        Args:
            y_pred: Predicted tensor array.
            y_targ: Target tensor array.

        Returns:
            The computed F1-Score as a float.
        """

        target_classes = self.model.output_shape[0]

        self.f1.fill(0)
        self.local_f1.fill(0)

        target_classes = np.int32(target_classes)

        self.kernel(
            self.f1.ary,
            self.conf_matrix_metric.conf_matrix.ary,
            self.local_f1.ary,
            target_classes,
            grid=self.grid,
            block=self.block,
            stream=self.model.pycuda_stream,
        )

        return self.f1.get().item()
