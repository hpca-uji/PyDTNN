"""PyCUDA implementation of the Binary Cross Entropy loss function."""

import logging

import numpy as np
from pycuda import gpuarray  # pyright: ignore[reportAttributeAccessIssue]

from pydtnn.backends.pycuda.losses.abstract.loss import LossPycuda
from pydtnn.backends.pycuda.utils.tensor_array import TensorArray
from pydtnn.losses.binary_cross_entropy import BinaryCrossEntropy

__all__ = ("BinaryCrossEntropyPycuda",)

logger = logging.getLogger(__name__)


class BinaryCrossEntropyPycuda(LossPycuda, BinaryCrossEntropy[TensorArray]):
    """PyCUDA-accelerated Binary Cross Entropy loss implementation."""

    def _model_init(self) -> None:
        """Initializes GPU memory buffers and model-dependent parameters."""
        super()._model_init()
        # NOTE: the model must be executed before this one.
        self.argmax = gpuarray.zeros((self.model.batch_size,), np.dtype(np.int32))
        self.memory_used += self.argmax.nbytes

    def compute(self, y_pred: TensorArray, y_targ: TensorArray) -> tuple[float, TensorArray]:
        """
        Computes the binary cross entropy loss and its gradient on the GPU.

        Args:
            y_pred: Predicted values from the model.
            y_targ: Target ground truth values.

        Returns:
            A tuple containing the scalar loss value and the gradient TensorArray.
        """
        batch_size = self.model.real_batch_size

        assert len(y_targ.shape) == 2
        self.kernel(
            y_targ,
            y_pred,
            self.loss,
            self.dx.ary,
            self.weights,
            self.argmax,
            batch_size,
            self.shape[1],
            self.eps,
            grid=self.grid,
            block=self.block,
            stream=self.model.stream,
        )
        sum_weights: float = gpuarray.sum(self.argmax[:batch_size]).get()

        loss = -gpuarray.sum(self.loss[:batch_size]).get() / sum_weights
        self.dx /= sum_weights

        return loss.item(), self.dx
