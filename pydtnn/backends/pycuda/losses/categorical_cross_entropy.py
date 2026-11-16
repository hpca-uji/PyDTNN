"""PyCUDA implementation of the Categorical Cross-Entropy loss function."""

import logging

import numpy as np
from pycuda import gpuarray  # pyright: ignore[reportAttributeAccessIssue]

from pydtnn.backends.pycuda.losses.abstract.loss import LossPycuda
from pydtnn.losses.categorical_cross_entropy import CategoricalCrossEntropy
from pydtnn.utils.tensor_array import DescriptorType, MemoryType, TensorArray

__all__ = ("CategoricalCrossEntropyPycuda",)

logger = logging.getLogger(__name__)


class CategoricalCrossEntropyPycuda(CategoricalCrossEntropy[TensorArray], LossPycuda):
    """Categorical Cross-Entropy loss implementation for PyCUDA backends."""

    def _model_init(self) -> None:
        """Initializes GPU memory buffers and model-dependent parameters."""
        super()._model_init()
        # NOTE: the model must be executed before this one.
        self.argmax = TensorArray.zeros(
            (self.model.batch_size,),
            self.model.dtype,
            tensor_format=self.model.tensor_format,
            mem_type=MemoryType.GLOBAL if self.model.use_gpudirect else MemoryType.DEVICE,
            desc_type=DescriptorType.NONE
        )
        self.memory_used += self.argmax.nbytes

    def compute(self, y_pred: TensorArray, y_targ: TensorArray) -> tuple[float, TensorArray]:
        """
        Computes the categorical cross-entropy loss and gradients on the GPU.

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
            self.weights,
            self.dx.ary,
            self.argmax,
            np.int32(batch_size),
            np.int32(self.shape[1]),
            np.float32(self.eps),
            gpuarray.sum(y_targ.ary).get(),
            grid=self.grid,
            block=self.block,
            stream=self.model.pycuda_stream,
        )

        sum_weights: float = gpuarray.sum(self.argmax[:batch_size].ary).get()

        loss = -gpuarray.sum(self.loss[:batch_size].ary).get() / sum_weights
        self.dx /= sum_weights

        return loss.item(), self.dx
