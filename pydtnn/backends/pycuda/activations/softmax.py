"""PyCUDA backend implementation for the Softmax activation layer."""

import logging
from typing import Any

from pydtnn.activations.softmax import Softmax
from pydtnn.backends.pycuda.activations.abstract.activation import ActivationPycuda
from pydtnn.libs import cudnn
from pydtnn.utils.constants import ArrayShape
from pydtnn.utils.tensor_array import FormatType, MemoryType, TensorArray

__all__ = ("SoftmaxPycuda",)

logger = logging.getLogger(__name__)


class SoftmaxPycuda(Softmax[TensorArray], ActivationPycuda):
    """PyCUDA-accelerated Softmax activation layer using cuDNN."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Initialize the SoftmaxPycuda layer."""
        super().__init__(*args, **kwargs)
        self.mode: int = None  # pyright: ignore[reportAttributeAccessIssue]
        self.algo: int = None  # pyright: ignore[reportAttributeAccessIssue]

    def _model_init(self, prev_shape: ArrayShape, x: TensorArray) -> None:
        """Initialize layer buffers and cuDNN parameters."""
        super()._model_init(prev_shape, x)

        self.mode = cudnn.cudnnSoftmaxMode["CUDNN_SOFTMAX_MODE_INSTANCE"]
        self.algo = cudnn.cudnnSoftmaxAlgorithm["CUDNN_SOFTMAX_ACCURATE"]

        # Activations y
        self.y = TensorArray.zeros(
            x.shape, self.model.dtype,
            tensor_format=self.model.tensor_format,
            mem_type=MemoryType.GLOBAL if self.model.use_gpudirect else MemoryType.DEVICE,
            fmt_type=FormatType.TENSOR
        )

        # Derivative dx
        self.dx = TensorArray.zeros(
            x.shape, self.model.dtype,
            tensor_format=self.model.tensor_format,
            mem_type=MemoryType.GLOBAL if self.model.use_gpudirect else MemoryType.DEVICE,
            fmt_type=FormatType.TENSOR
        )

        self.memory_used += self.y.nbytes + self.dx.nbytes

    def forward(self, x: TensorArray) -> TensorArray:
        """Perform the forward pass using cuDNN."""
        alpha, beta = 1.0, 0.0
        cudnn.cudnnSoftmaxForward(
            self.model.cudnn_handle,
            self.algo,
            self.mode,
            alpha,
            x.desc,
            x.ptr,
            beta,
            self.y.desc,
            self.y.ptr,
        )
        return self.y

    def backward(self, dy: TensorArray) -> TensorArray:
        """Perform the backward pass using cuDNN."""
        alpha, beta = 1.0, 0.0
        cudnn.cudnnSoftmaxBackward(
            self.model.cudnn_handle,
            self.algo,
            self.mode,
            alpha,
            self.y.desc,
            self.y.ptr,
            dy.desc,
            dy.ptr,
            beta,
            self.dx.desc,
            self.dx.ptr,
        )
        return self.dx
