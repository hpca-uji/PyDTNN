"""PyCUDA backend implementation for the Arctanh activation function."""

import logging

import numpy as np
from pycuda import gpuarray  # type: ignore
from pycuda.elementwise import ElementwiseKernel  # type: ignore

from pydtnn.activations.arctanh import Arctanh
from pydtnn.backends.pycuda.activations.abstract.activation import ActivationPycuda
from pydtnn.backends.pycuda.utils.tensor_array import TensorArray
from pydtnn.utils.constants import DTYPE2CTYPE, ArrayShape

__all__ = ("ArctanhPycuda",)

logger = logging.getLogger(__name__)


class ArctanhPycuda(Arctanh[TensorArray], ActivationPycuda):
    """PyCUDA implementation of the Arctanh activation layer."""

    def __init__(self, *args, **kwargs):
        """Initialize the ArctanhPycuda layer."""
        super().__init__(*args, **kwargs)
        self.atanh: ElementwiseKernel = None
        self.datanh: ElementwiseKernel = None

    def _model_init(self, prev_shape: ArrayShape, x: TensorArray) -> None:
        """Initialize kernels and allocate memory for forward and backward passes."""
        super()._model_init(prev_shape, x)

        self.atanh = ElementwiseKernel(
            "{T} *in, {T} *out".format(T=DTYPE2CTYPE[self.model.dtype]),
            "out[i] = {func}(in[i]);".format(func={np.dtype(np.float32): "atanhf", np.dtype(np.float64): "atanh"}[self.model.dtype]),
            "k_atanh",
        )

        self.datanh = ElementwiseKernel(
            "{T} *in, {T} *out".format(T=DTYPE2CTYPE[self.model.dtype]),
            "out[i] = 1.0 / (1.0 + {func}(in[i], 2));".format(func={np.dtype(np.float32): "powf", np.dtype(np.float64): "pow"}[self.model.dtype]),
            "datanh",
        )

        # Activations y
        y_gpu = gpuarray.zeros(x.shape, self.model.dtype)
        self.y = TensorArray(y_gpu, self.model.tensor_format, self.model.cudnn_dtype)

        # Derivative dx
        dx_gpu = gpuarray.zeros(x.shape, self.model.dtype)
        self.dx = TensorArray(dx_gpu, self.model.tensor_format, self.model.cudnn_dtype)

        self.memory_used += self.y.nbytes + self.dx.nbytes

    def forward(self, x: TensorArray) -> TensorArray:
        """Perform the forward pass using the PyCUDA atanh kernel."""
        self.atanh(x.ary, self.y, stream=self.model.stream)
        return self.y

    def backward(self, dy: TensorArray) -> TensorArray:
        """Perform the backward pass using the PyCUDA datanh kernel."""
        # Compute dx
        self.datanh(dy.ary, self.dx.ary, stream=self.model.stream)
        return self.dx
