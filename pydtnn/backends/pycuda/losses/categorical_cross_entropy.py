"""
PyCUDA implementation of the Categorical Cross-Entropy loss function.
"""
import logging

import numpy as np
from pycuda import gpuarray  # type: ignore

from pydtnn.backends.pycuda.losses.loss import LossPycuda
from pydtnn.backends.pycuda.utils.tensor_array import TensorArray
from pydtnn.losses.categorical_cross_entropy import CategoricalCrossEntropy

__all__ = ("CategoricalCrossEntropyPycuda",)

logger = logging.getLogger(__name__)


class CategoricalCrossEntropyPycuda(LossPycuda, CategoricalCrossEntropy[TensorArray]):
    """
    Categorical Cross-Entropy loss implementation for PyCUDA backends.
    """
    def compute(self, y_pred: TensorArray, y_targ: TensorArray, batch_size: int) -> tuple[float, TensorArray]:
        """
        Computes the categorical cross-entropy loss and gradients on the GPU.

        Args:
            y_pred: Predicted probabilities from the model.
            y_targ: Ground truth labels.
            batch_size: Number of samples in the current batch.

        Returns:
            A tuple containing the scalar loss value and the gradient tensor.
        """
        self.kernel(y_targ.ary, y_pred.ary, self.loss, self.dx.ary, np.int32(batch_size), np.int32(self.shape[1]), np.float32(self.eps), grid=self.grid, block=self.block, stream=self.model.stream)
        loss = -gpuarray.sum(self.loss[:batch_size]).get() / batch_size
        return float(loss), self.dx