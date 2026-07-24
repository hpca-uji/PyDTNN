"""PyCUDA backend implementation for the Tanh activation function."""

import logging
from typing import Any

from pycuda import gpuarray  # pyright: ignore[reportAttributeAccessIssue]

from pydtnn.activations.tanh import Tanh
from pydtnn.backends.pycuda.activations.abstract.activation import ActivationPycuda
from pydtnn.backends.pycuda.utils.tensor_array import TensorArray
from pydtnn.libs import cudnn as cudnn
from pydtnn.utils.constants import ArrayShape

__all__ = ("TanhPycuda",)

logger = logging.getLogger(__name__)


class TanhPycuda(Tanh[TensorArray], ActivationPycuda):
    """PyCUDA implementation of the Tanh activation layer using cuDNN."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Initialize the TanhPycuda layer."""
        super().__init__(*args, **kwargs)
        self.act_desc: int = None  # pyright: ignore[reportAttributeAccessIssue]

    def _model_init(self, prev_shape: ArrayShape, x: TensorArray) -> None:
        """Initialize cuDNN activation descriptor and allocate GPU memory for buffers."""
        super()._model_init(prev_shape, x)

        self.act_desc = cudnn.cudnnCreateActivationDescriptor()

        mode = cudnn.cudnnActivationMode["CUDNN_ACTIVATION_TANH"]
        nan = cudnn.cudnnNanPropagation["CUDNN_NOT_PROPAGATE_NAN"]
        cudnn.cudnnSetActivationDescriptor(self.act_desc, mode, nan, 0.0)

        # Activations y
        y_gpu = gpuarray.zeros(x.ary.shape, self.model.dtype)
        self.y = TensorArray(y_gpu, self.model.tensor_format, self.model.cudnn_dtype)

        # Derivative dx
        dx_gpu = gpuarray.zeros(x.ary.shape, self.model.dtype)
        self.dx = TensorArray(dx_gpu, self.model.tensor_format, self.model.cudnn_dtype)

        self.memory_used += self.y.nbytes + self.dx.nbytes

    def forward(self, x: TensorArray) -> TensorArray:
        """Perform the forward pass using cuDNN activation."""
        alpha, beta = 1.0, 0.0
        cudnn.cudnnActivationForward(
            self.model.cudnn_handle,
            self.act_desc,
            alpha,
            x.desc,
            x.ptr_voidp,
            beta,
            self.y.desc,
            self.y.ptr_voidp,
        )
        return self.y

    def backward(self, dy: TensorArray) -> TensorArray:
        """Perform the backward pass using cuDNN activation."""
        alpha, beta = 1.0, 0.0
        cudnn.cudnnActivationBackward(
            self.model.cudnn_handle,
            self.act_desc,
            alpha,
            self.y.desc,
            self.y.ptr_voidp,
            dy.desc,
            dy.ptr_voidp,
            self.x.desc,
            self.x.ptr_voidp,
            beta,
            self.dx.desc,
            self.dx.ptr_voidp,
        )
        return self.dx
