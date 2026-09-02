"""PyCUDA implementation of the Categorical Cross-Entropy loss function."""

import logging

import numpy as np
from pycuda import gpuarray  # pyright: ignore[reportAttributeAccessIssue]

from pydtnn.backends.pycuda.losses.abstract.loss import LossPycuda
from pydtnn.backends.pycuda.utils.tensor_array import TensorArray
from pydtnn.losses.cross_entropy import CrossEntropy

__all__ = ("CrossEntropyPycuda",)

logger = logging.getLogger(__name__)


class CrossEntropyPycuda(CrossEntropy[TensorArray], LossPycuda):
    """Cross Entropy loss implementation for PyCUDA backends."""

    def _model_init(self) -> None:
        """Initialize GPU memory buffers and model-dependent parameters."""
        super()._model_init()
        self.argmax = gpuarray.zeros((self.model.batch_size,), np.dtype(self.model.dtype))
        self.sample_weights = gpuarray.zeros((self.model.batch_size,), np.dtype(self.model.dtype))
        self.memory_used += self.argmax.nbytes
        self.memory_used += self.sample_weights.nbytes

    def compute(self, y_pred: TensorArray, y_targ: TensorArray) -> tuple[float, TensorArray]:
        """
        Compute the cross entropy loss and gradients.

        Args:
            y_pred: Logits predicted by the model.
            y_targ: Ground truth labels in one-hot encoded format.

        Returns:
            A tuple containing the scalar loss value and the gradient tensor.
        """
        batch_size = self.model.real_batch_size

        self.kernel(
            y_targ.ary,
            y_pred.ary,
            self.loss,
            self.weights,
            self.dx.ary,
            self.argmax,
            self.sample_weights,
            np.int32(batch_size),
            np.int32(self.shape[1]),
            grid=self.grid,
            block=self.block,
            stream=self.model.stream,
        )

        sum_weights = float(gpuarray.sum(self.sample_weights[:batch_size]).get())
        loss = -gpuarray.sum(self.loss[:batch_size]).get() / sum_weights
        self.dx /= sum_weights

        return loss.item(), self.dx
