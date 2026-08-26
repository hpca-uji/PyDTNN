"""PyCUDA implementation of the Categorical Cross-Entropy loss function."""

import logging

import numpy as np
from pycuda import gpuarray  # pyright: ignore[reportAttributeAccessIssue]

from pydtnn.backends.pycuda.losses.abstract.loss import LossPycuda
from pydtnn.backends.pycuda.utils.tensor_array import TensorArray
from pydtnn.losses.negative_likelihood import NegativeLikelihood

__all__ = ("NegativeLikelihoodPycuda",)

logger = logging.getLogger(__name__)


class NegativeLikelihoodPycuda(NegativeLikelihood[TensorArray], LossPycuda):
    """Negative Likelihood loss implementation for PyCUDA backends."""

    def _model_init(self) -> None:
        """Initializes GPU memory buffers and model-dependent parameters."""
        super()._model_init()
        # NOTE: the model must be executed before this one.
        self.argmax: gpuarray.GPUArray = gpuarray.zeros((self.model.batch_size,), np.dtype(self.model.dtype))
        self.memory_used += self.argmax.nbytes

    def compute(self, y_pred: TensorArray, y_targ: TensorArray) -> tuple[float, TensorArray]:
        """
        Computes the Negative Likelihood loss and gradients on the GPU.

        Args:
            y_pred: Predicted probabilities from the model.
            y_targ: Ground truth labels.

        Returns:
            A tuple containing the scalar loss value and the gradient tensor.
        """
        batch_size = self.model.real_batch_size

        self.kernel(
            y_targ.ary,
            y_pred.ary,
            self.loss,
            self.weights.ary,
            self.dx.ary,
            self.argmax,
            np.int32(batch_size),
            np.int32(self.shape[1]),
            grid=self.grid,
            block=self.block,
            stream=self.model.stream,
        )
        sum_weights = float(gpuarray.sum(self.argmax[:batch_size]).get())

        loss = -gpuarray.sum(self.loss[:batch_size]).get() / sum_weights
        self.dx /= sum_weights

        return loss.item(), self.dx
