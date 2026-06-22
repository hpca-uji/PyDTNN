"""
PyCUDA implementation of the Categorical Cross-Entropy loss function.
"""

import logging

import numpy as np
from pycuda import gpuarray  # type: ignore

from pydtnn.backends.pycuda.losses.abstract.loss import LossPycuda
from pydtnn.backends.pycuda.utils.tensor_array import TensorArray
from pydtnn.losses.categorical_cross_entropy import CategoricalCrossEntropy

__all__ = ("CategoricalCrossEntropyPycuda",)

logger = logging.getLogger(__name__)


class CategoricalCrossEntropyPycuda(LossPycuda, CategoricalCrossEntropy[TensorArray]):
    """
    Categorical Cross-Entropy loss implementation for PyCUDA backends.
    """

    def compute(
        self, y_pred: TensorArray, y_targ: TensorArray, batch_size: int
    ) -> tuple[float, TensorArray]:
        """
        Computes the categorical cross-entropy loss and gradients on the GPU.

        Args:
            y_pred: Predicted probabilities from the model.
            y_targ: Ground truth labels.
            batch_size: Number of samples in the current batch.

        Returns:
            A tuple containing the scalar loss value and the gradient tensor.
        """
        loss2 = self.loss.copy()

        self.kernel(
            y_targ.ary,
            y_pred.ary,
            self.loss,
            self.dx.ary,
            np.int32(batch_size),
            np.int32(self.shape[1]),
            np.float32(self.eps),
            grid=self.grid,
            block=self.block,
            stream=self.model.stream,
        )

        kernel2 = self._get_kernel(code_file_name="categorical_cross_entropy_test.cu")
        kernel2(
            y_targ.ary,
            y_pred.ary,
            loss2,
            self.weights,
            self.dx.ary,
            np.int32(batch_size),
            np.int32(self.shape[1]),
            np.float32(self.eps),
            grid=self.grid,
            block=self.block,
            stream=self.model.stream,
        )
        breakpoint()
        l1 = self.loss.get()
        l2 = loss2.get()
        diff = l1 - l2
        breakpoint()

        loss = -gpuarray.sum(self.loss[:batch_size]).get() / batch_size
        return loss.item(), self.dx
