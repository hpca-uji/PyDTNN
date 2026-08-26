"""PyCUDA implementation of the Log activation function."""

import logging
import math
from typing import Any

import numpy as np
from pycuda import gpuarray  # pyright: ignore[reportAttributeAccessIssue]

from pydtnn.activations.log import Log
from pydtnn.backends.pycuda.activations.abstract.activation import ActivationPycuda
from pydtnn.backends.pycuda.utils.tensor_array import TensorArray
from pydtnn.tracers.events import PYDTNN_OPS_EVENT, PYDTNN_OPS_EVENTS, OpsEventEnum
from pydtnn.utils.constants import DTYPE2CTYPE, ArrayShape

__all__ = ("LogPycuda",)

logger = logging.getLogger(__name__)


class LogPycuda(Log[TensorArray], ActivationPycuda):
    """PyCUDA-accelerated Log activation layer."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Initialize the LogPycuda layer."""
        super().__init__(*args, **kwargs)

        self.y: TensorArray = None  # pyright: ignore[reportAttributeAccessIssue]
        self.dx: TensorArray = None  # pyright: ignore[reportAttributeAccessIssue]

    def _model_init(self, prev_shape: ArrayShape, x: TensorArray) -> None:
        """Initialize model parameters, CUDA kernels, and memory buffers."""
        super()._model_init(prev_shape, x)

        # Activation output y = log(x)
        y_gpu = gpuarray.zeros((self.model.batch_size, *self.shape), self.model.dtype)
        self.y = TensorArray(y_gpu, self.model.tensor_format, self.model.cudnn_dtype)

        # Derivative dx
        dx_gpu = gpuarray.zeros((self.model.batch_size, *self.shape), self.model.dtype)
        self.dx = TensorArray(dx_gpu, self.model.tensor_format, self.model.cudnn_dtype,)

        self.memory_used += self.y.nbytes + self.dx.nbytes

        self.defines_replaces = {
            '"TYPE"': DTYPE2CTYPE[self.model.dtype],
        }

        self.cuda_fwd_func = self._fwd_kernel()
        self.cuda_bwd_func = self._bwd_kernel()

        self.total_num_threads = np.int32(
            math.prod(self.grid) * math.prod(self.block)
        )

    def forward(self, x: TensorArray) -> TensorArray:
        """Perform the forward pass of the Log activation."""
        n = np.int32(math.prod(x.shape))

        self.cuda_fwd_func(
            x.ary,
            self.y.ary,
            self.total_num_threads,
            n,
            grid=self.grid,
            block=self.block,
            stream=self.model.stream,
        )

        return self.y

    def backward(self, dy: TensorArray) -> TensorArray:
        """Perform the backward pass of the Log activation."""
        n = np.int32(math.prod(dy.shape))

        self.cuda_bwd_func(
            self.dx.ary,
            dy.ary,
            self.y.ary,
            self.total_num_threads,
            n,
            grid=self.grid,
            block=self.block,
            stream=self.model.stream,
        )

        return self.dx
