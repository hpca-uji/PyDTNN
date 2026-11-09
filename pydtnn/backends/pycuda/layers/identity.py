"""PyCUDA implementation of the Input layer for the PyDTNN framework."""

import logging

from pycuda import gpuarray  # pyright: ignore[reportAttributeAccessIssue]

from pydtnn.backends.pycuda.layers.abstract.layer import LayerPycuda
from pydtnn.layers.identity import Identity
from pydtnn.utils.constants import ArrayShape
from pydtnn.utils.tensor_array import TensorArray

__all__ = ("IdentityPycuda",)

logger = logging.getLogger(__name__)


class IdentityPycuda(Identity[TensorArray], LayerPycuda):
    """PyCUDA-specific implementation of the identity layer."""

    def _model_init(self, prev_shape: ArrayShape, x: TensorArray) -> None:
        """Initialize layer memory and GPU buffers."""
        super()._model_init(prev_shape, x)

        y_gpu = gpuarray.zeros((self.model.batch_size, *self.shape), self.model.dtype)
        self.y = TensorArray(y_gpu, self.model.tensor_format, self.model.cudnn_dtype)
        self.memory_used += self.y.nbytes

    def forward(self, x: TensorArray) -> TensorArray:
        """Perform forward pass."""
        return x

    def backward(self, dy: TensorArray) -> TensorArray:
        """Perform backward pass."""
        return dy

    def _state_reduce_async_nccl(self, state_: str) -> None:
        self.model.pycuda_stream.synchronize()
        super()._state_reduce_async_nccl(state_)

    def _state_reduce_sync_nccl(self, state_: str) -> None:
        self.model.pycuda_stream.synchronize()
        super()._state_reduce_sync_nccl(state_)

    def _state_reduce_async_cpu(self, state_: str) -> None:
        self.model.pycuda_stream.synchronize()
        super()._state_reduce_async_cpu(state_)

    def _state_reduce_sync_cpu(self, state_: str) -> None:
        self.model.pycuda_stream.synchronize()
        super()._state_reduce_sync_cpu(state_)

    def _state_reduce_async(self, state_: str) -> None:
        self.model.pycuda_stream.synchronize()
        super()._state_reduce_async(state_)

    def _state_reduce_sync(self, state_: str) -> None:
        self.model.pycuda_stream.synchronize()
        super()._state_reduce_sync(state_)
