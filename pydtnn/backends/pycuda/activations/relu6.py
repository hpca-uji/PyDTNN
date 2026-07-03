"""PyCUDA implementation of the ReLU6 activation function."""

import logging
import math
from typing import Any

import numpy as np
from pycuda import gpuarray  # type: ignore

from pydtnn.activations.relu6 import Relu6
from pydtnn.backends.pycuda.activations.abstract.activation import ActivationPycuda
from pydtnn.backends.pycuda.utils.tensor_array import TensorArray
from pydtnn.tracers.events import PYDTNN_OPS_EVENT, PYDTNN_OPS_EVENTS, OpsEventEnum
from pydtnn.utils.constants import DTYPE2CTYPE, ArrayShape
from pydtnn.utils.performance_models import col2im_time, im2col_time

__all__ = ("Relu6Pycuda",)

logger = logging.getLogger(__name__)


class Relu6Pycuda(Relu6[TensorArray], ActivationPycuda):
    """PyCUDA-accelerated ReLU6 activation layer."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Initialize the Relu6Pycuda layer."""
        super().__init__(*args, **kwargs)
        self.mask: TensorArray = None  # type: ignore
        self.y: TensorArray = None  # type: ignore

    def _model_init(self, prev_shape: ArrayShape, x: TensorArray) -> None:
        """Initialize model parameters, CUDA kernels, and memory buffers."""
        super()._model_init(prev_shape, x)

        y_gpu = gpuarray.zeros(x.shape, self.model.dtype)
        self.y = TensorArray(y_gpu, self.model.tensor_format, self.model.cudnn_dtype)

        mask_gpu = gpuarray.zeros((self.model.batch_size, *self.prev_shape), self.model.dtype)
        self.mask = TensorArray(mask_gpu, self.model.tensor_format, self.model.cudnn_dtype)

        self.memory_used += self.y.nbytes + self.mask.nbytes

        self.defines_replaces = {'"TYPE"': DTYPE2CTYPE[self.model.dtype]}
        self.cuda_fwd_func = self._fwd_kernel()
        self.cuda_bwd_func = self._bwd_kernel()

        self.total_num_threads = np.int32(math.prod(self.grid) * math.prod(self.block))

        self.initialize_relu_2d_gpu(prev_shape)

    def forward(self, x: TensorArray) -> TensorArray:
        """Perform the forward pass of the ReLU6 activation."""
        self.model.tracer.emit_event(
            PYDTNN_OPS_EVENT, self.id * PYDTNN_OPS_EVENTS + OpsEventEnum.FORWARD_CUDNN
        )

        n = np.int32(math.prod(x.shape))

        self.cuda_fwd_func(
            x.ary,
            self.mask.ary,
            self.max.ary,
            np.float32(self.cap),
            self.total_num_threads,
            n,
            grid=self.grid,
            block=self.block,
            stream=self.model.stream,
        )

        self.y: TensorArray = self.mask

        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, 0)

        return self.y

    def backward(self, dy: TensorArray) -> TensorArray:
        """Perform the backward pass of the ReLU6 activation."""
        self.model.tracer.emit_event(
            PYDTNN_OPS_EVENT, self.id * PYDTNN_OPS_EVENTS + OpsEventEnum.BACKWARD_CUDNN_DX
        )

        n = np.int32(math.prod(dy.shape))

        self.cuda_bwd_func(
            self.dx.ary,
            dy.ary,
            self.mask.ary,
            self.total_num_threads,
            n,
            grid=self.grid,
            block=self.block,
            stream=self.model.stream,
        )

        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, 0)
        return self.dx

    def initialize_relu_2d_gpu(self, prev_shape: ArrayShape) -> None:
        """Initialize GPU buffers and performance models for 2D ReLU operations."""
        self.ci, self.hi, self.wi = self.model.decode_shape(prev_shape)
        self.shape = prev_shape

        n: int = self.model.batch_size * self.hi * self.wi * self.ci

        _max = gpuarray.zeros((self.model.batch_size, *self.shape), self.model.dtype)
        self.max = TensorArray(_max, self.model.tensor_format, self.model.cudnn_dtype)

        _mask = gpuarray.zeros((self.model.batch_size, *self.shape), self.model.dtype)
        self.mask = TensorArray(_mask, self.model.tensor_format, self.model.cudnn_dtype)

        # Derivative dx
        dx_gpu = gpuarray.zeros((self.model.batch_size, *self.shape), self.model.dtype)
        self.dx = TensorArray(dx_gpu, self.model.tensor_format, self.model.cudnn_dtype)

        self.fwd_time = im2col_time(
            m=self.ci,
            n=n,
            cpu_speed=self.model.cpu_speed,
            memory_bw=self.model.memory_bw,
            dtype=self.model.dtype,
        )  # type: ignore (it's fine)
        self.bwd_time = col2im_time(
            m=self.ci,
            n=n,
            cpu_speed=self.model.cpu_speed,
            memory_bw=self.model.memory_bw,
            dtype=self.model.dtype,
        )  # type: ignore (it's fine)
