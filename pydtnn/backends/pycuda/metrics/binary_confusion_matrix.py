"""PyCUDA implementation of the binary confusion matrix metric."""

import logging

import numpy as np

from pydtnn.backends.pycuda.metrics.abstract.metric import MetricPycuda
from pydtnn.metrics.binary_confusion_matrix import BinaryConfusionMatrix
from pydtnn.utils.tensor_array import FormatType, MemoryType, TensorArray

__all__ = ("BinaryConfusionMatrixPycuda",)

logger = logging.getLogger(__name__)


class BinaryConfusionMatrixPycuda(BinaryConfusionMatrix[TensorArray], MetricPycuda):
    """PyCUDA-accelerated binary confusion matrix calculation."""

    def _model_init(self) -> None:
        """Initializes the confusion matrix buffers on the GPU."""
        super()._model_init()
        n = self.model.batch_size
        target_classes = self.model.output_shape[0]

        self.conf_matrix = TensorArray[tuple[int, int, int, int], np.dtype[np.int32]].zeros(
            (1, target_classes, 2, 2),
            np.dtype(np.int32),
            tensor_format=self.model.tensor_format,
            mem_type=MemoryType.GLOBAL if self.model.use_gpudirect else MemoryType.DEVICE,
            fmt_type=FormatType.NONE
        )
        self.local_cm = TensorArray[tuple[int, int, int, int], np.dtype[np.int32]].zeros(
            (n, target_classes, 2, 2),
            np.dtype(np.int32),
            tensor_format=self.model.tensor_format,
            mem_type=MemoryType.GLOBAL if self.model.use_gpudirect else MemoryType.DEVICE,
            fmt_type=FormatType.NONE
        )

    def compute(self, y_pred: TensorArray, y_targ: TensorArray) -> np.ndarray:
        """
        Computes the confusion matrix for the given predictions and targets.

        For every label in target class, there is one confusion matrix like this:
                |Predicted|
        ________| T  | F  |
        Target|T| TP | FN |
              |F| FP | TN |
        """

        n = self.model.real_batch_size
        target_classes = self.model.output_shape[0]

        self.conf_matrix.fill(0)
        self.local_cm.fill(0)

        n = np.int32(n)
        num_classes = np.int32(target_classes)
        self.kernel(
            y_targ.ary,
            y_pred.ary,
            self.conf_matrix.ary,
            self.local_cm.ary,
            num_classes,
            n,
            grid=self.grid,
            block=self.block,
            stream=self.model.pycuda_stream,
        )

        return self.conf_matrix.get()
