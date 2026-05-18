"""
PyCUDA implementation of the binary confusion matrix metric.
"""

import logging

import numpy as np

from pydtnn.backends.pycuda.metrics.abstract.metric import MetricPycuda
from pydtnn.backends.pycuda.utils.tensor_array import TensorArray
from pydtnn.metrics.binary_confusion_matrix import BinaryConfusionMatrix

__all__ = ("BinaryConfusionMatrixPycuda",)

logger = logging.getLogger(__name__)


class BinaryConfusionMatrixPycuda(BinaryConfusionMatrix[TensorArray], MetricPycuda):
    """
    PyCUDA-accelerated binary confusion matrix calculation.
    """

    def _model_init(self) -> None:
        """
        Initializes the confusion matrix buffers on the GPU.
        """
        super()._model_init()
        n = self.model.batch_size
        target_classes = self.model.output_shape[0]

        self.conf_matrix = TensorArray.new_zeros(shape=(1, target_classes, 2, 2), dtype=np.dtype(np.int32), tensor_format=self.model.tensor_format, cudnn_dtype=self.model.cudnn_dtype)
        self.local_cm = TensorArray.new_zeros(shape=(n, target_classes, 2, 2), dtype=np.dtype(np.int32), tensor_format=self.model.tensor_format, cudnn_dtype=self.model.cudnn_dtype)

    def compute(self, y_pred: TensorArray, y_targ: TensorArray) -> np.ndarray:
        """
        Computes the confusion matrix for the given predictions and targets.

        For every label in target class, there is one confusion matrix like this:
                |Predicted|
        ________| T  | F  |
        Target|T| TP | FN |
              |F| FP | TN |
        """

        n = self.model.batch_size
        target_classes = self.model.output_shape[0]

        self.conf_matrix.fill(0)
        self.local_cm.fill(0)

        n = np.int32(n)
        num_classes = np.int32(target_classes)
        self.kernel(y_targ.ary, y_pred.ary, self.conf_matrix.ary, self.local_cm.ary, num_classes, n, grid=self.grid, block=self.block, stream=self.model.stream)

        return self.conf_matrix.get()
