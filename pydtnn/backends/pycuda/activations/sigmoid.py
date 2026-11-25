"""PyCUDA backend implementation of the Sigmoid activation function."""

import logging
from typing import Any

from pydtnn.activations.sigmoid import Sigmoid
from pydtnn.backends.pycuda.activations.abstract.activation import ActivationPycuda
from pydtnn.libs import cudnn
from pydtnn.utils.constants import ArrayShape
from pydtnn.utils.tensor_array import FormatType, MemoryType, TensorArray

__all__ = ("SigmoidPycuda",)

logger = logging.getLogger(__name__)


class SigmoidPycuda(Sigmoid[TensorArray], ActivationPycuda):
    """PyCUDA-accelerated Sigmoid activation layer."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Initialize the SigmoidPycuda layer."""
        super().__init__(*args, **kwargs)
        self.act_desc: int = None  # pyright: ignore[reportAttributeAccessIssue]

    def _model_init(self, prev_shape: ArrayShape, x: TensorArray) -> None:
        """Initialize cuDNN descriptors and allocate memory for forward and backward passes."""
        super()._model_init(prev_shape, x)

        self.act_desc = cudnn.cudnnCreateActivationDescriptor()

        mode = cudnn.cudnnActivationMode["CUDNN_ACTIVATION_SIGMOID"]
        nan = cudnn.cudnnNanPropagation["CUDNN_NOT_PROPAGATE_NAN"]
        cudnn.cudnnSetActivationDescriptor(self.act_desc, mode, nan, 0.0)

        # Activations a
        self.y = TensorArray.zeros(
            x.shape,
            self.model.dtype,
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
        cudnn.cudnnActivationForward(
            self.model.cudnn_handle,
            self.act_desc,
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
        cudnn.cudnnActivationBackward(
            self.model.cudnn_handle,
            self.act_desc,
            alpha,
            self.y.desc,
            self.y.ptr,
            dy.desc,
            dy.ptr,
            self.x.desc,
            self.x.ptr,
            beta,
            self.dx.desc,
            self.dx.ptr,
        )
        return self.dx
