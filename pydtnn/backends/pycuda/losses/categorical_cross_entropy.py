"""PyCUDA implementation of the Categorical Cross-Entropy loss function."""

import logging

import numpy as np
from pycuda import gpuarray  # pyright: ignore[reportAttributeAccessIssue]

from pydtnn.backends.pycuda.losses.abstract.loss import LossPycuda
from pydtnn.backends.pycuda.utils.tensor_array import TensorArray
from pydtnn.losses.categorical_cross_entropy import CategoricalCrossEntropy

__all__ = ("CategoricalCrossEntropyPycuda",)

logger = logging.getLogger(__name__)


class CategoricalCrossEntropyPycuda(CategoricalCrossEntropy[TensorArray], LossPycuda):
    """Categorical Cross-Entropy loss implementation for PyCUDA backends."""

    def _model_init(self) -> None:
        """Initializes GPU memory buffers and model-dependent parameters."""
        super()._model_init()
        # NOTE: the model must be executed before this one.
        self.argmax = gpuarray.zeros((self.model.batch_size,), np.dtype(np.int32))
        self.memory_used += self.argmax.nbytes

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

        self.kernel(
            y_targ.ary,
            y_pred.ary,
            self.loss,
            self.weights,
            self.dx.ary,
            self.argmax,
            np.int32(batch_size),
            np.int32(self.shape[1]),
            np.float32(self.eps),
            gpuarray.sum(y_targ).get(),
            grid=self.grid,
            block=self.block,
            stream=self.model.stream,
        )

        sum_weights: float = gpuarray.sum(self.argmax[:batch_size]).get()

        loss = -gpuarray.sum(self.loss[:batch_size]).get() / sum_weights
        self.dx /= sum_weights

        return loss.item(), self.dx
