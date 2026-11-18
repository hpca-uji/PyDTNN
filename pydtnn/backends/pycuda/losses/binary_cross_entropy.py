"""PyCUDA implementation of the Binary Cross Entropy loss function."""

import logging

from pycuda import gpuarray  # pyright: ignore[reportAttributeAccessIssue]

from pydtnn.backends.pycuda.losses.abstract.loss import LossPycuda
from pydtnn.losses.binary_cross_entropy import BinaryCrossEntropy
from pydtnn.utils.tensor_array import FormatType, MemoryType, TensorArray

__all__ = ("BinaryCrossEntropyPycuda",)

logger = logging.getLogger(__name__)


class BinaryCrossEntropyPycuda(LossPycuda, BinaryCrossEntropy[TensorArray]):
    """PyCUDA-accelerated Binary Cross Entropy loss implementation."""

    def _model_init(self) -> None:
        """Initializes GPU memory buffers and model-dependent parameters."""
        super()._model_init()
        # NOTE: the model must be executed before this one.
        self.argmax = TensorArray.zeros(
            (self.model.batch_size,),
            self.model.dtype,
            tensor_format=self.model.tensor_format,
            mem_type=MemoryType.GLOBAL if self.model.use_gpudirect else MemoryType.DEVICE,
            fmt_type=FormatType.NONE
        )
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
            y_targ.ary,
            y_pred.ary,
            self.loss,
            self.dx.ary,
            self.weights.ary,
            self.argmax.ary,
            batch_size,
            self.shape[1],
            self.eps,
            grid=self.grid,
            block=self.block,
            stream=self.model.pycuda_stream,
        )
        sum_weights: float = gpuarray.sum(self.argmax[:batch_size].ary).get()

        loss = -gpuarray.sum(self.loss[:batch_size].ary).get() / sum_weights
        self.dx /= sum_weights

        return loss.item(), self.dx
