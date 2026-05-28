"""
PyCUDA implementation of the Binary Cross Entropy loss function.
"""

import logging

from pycuda import gpuarray  # type: ignore

from pydtnn.backends.pycuda.losses.abstract.loss import LossPycuda
from pydtnn.backends.pycuda.utils.tensor_array import TensorArray
from pydtnn.losses.binary_cross_entropy import BinaryCrossEntropy

__all__ = ("BinaryCrossEntropyPycuda",)

logger = logging.getLogger(__name__)


class BinaryCrossEntropyPycuda(LossPycuda, BinaryCrossEntropy[TensorArray]):
    """
    PyCUDA-accelerated Binary Cross Entropy loss implementation.
    """

    def compute(
        self, y_pred: TensorArray, y_targ: TensorArray, batch_size: int
    ) -> tuple[float, TensorArray]:
        """
        Computes the binary cross entropy loss and its gradient on the GPU.

        Args:
            y_pred: Predicted values from the model.
            y_targ: Target ground truth values.
            batch_size: Number of samples in the current batch.

        Returns:
            A tuple containing the scalar loss value and the gradient TensorArray.
        """

        assert len(y_targ.shape) == 2
        self.kernel(
            y_targ,
            y_pred,
            self.loss,
            self.dx.ary,
            batch_size,
            self.shape[1],
            self.eps,
            grid=self.grid,
            block=self.block,
            stream=self.model.stream,
        )
        loss = -gpuarray.sum(self.loss[:batch_size]).get() / batch_size
        return loss.item(), self.dx
