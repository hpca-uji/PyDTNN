"""PyCUDA implementation of the Recall metric for PyDTNN."""

import logging

import numpy as np

from pydtnn.backends.pycuda.metrics.abstract.metric import MetricPycuda
from pydtnn.metrics.recall import Recall
from pydtnn.utils.tensor_array import DescriptorType, MemoryType, TensorArray

__all__ = ("RecallPycuda",)

logger = logging.getLogger(__name__)


class RecallPycuda(Recall[TensorArray], MetricPycuda):
    """Recall metric implementation using PyCUDA for GPU acceleration."""

    def _model_init(self) -> None:
        """Initializes the internal buffers for recall calculation on the GPU."""
        super()._model_init()
        target_classes = self.model.output_shape[0]
        self.recall = TensorArray.zeros(
            (1,),
            self.model.dtype,
            tensor_format=self.model.tensor_format,
            mem_type=MemoryType.GLOBAL if self.model.use_gpudirect else MemoryType.DEVICE,
            desc_type=DescriptorType.TENSOR
        )
        self.memory_used += self.recall.nbytes
        self.local_recall = TensorArray.zeros(
            (target_classes,),
            self.model.dtype,
            tensor_format=self.model.tensor_format,
            mem_type=MemoryType.GLOBAL if self.model.use_gpudirect else MemoryType.DEVICE,
            desc_type=DescriptorType.TENSOR
        )
        self.memory_used += self.local_recall.nbytes

    def compute(self, y_pred: TensorArray, y_targ: TensorArray) -> float:
        """
        Computes the recall score using a PyCUDA kernel.

        Args:
            y_pred: Predicted tensor array.
            y_targ: Target tensor array.

        Returns:
            The computed recall value as a float.
        """

        target_classes = self.model.output_shape[0]

        self.recall.fill(0)
        self.local_recall.fill(0)

        target_classes = np.int32(target_classes)
        self.kernel(
            self.recall.ary,
            self.conf_matrix_metric.conf_matrix.ary,
            self.local_recall.ary,
            target_classes,
            grid=self.grid,
            block=self.block,
            stream=self.model.pycuda_stream,
        )

        return self.recall.get().item()
