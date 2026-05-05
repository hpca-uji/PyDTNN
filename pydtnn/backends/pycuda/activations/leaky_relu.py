import logging
import math

import numpy as np
from pycuda import gpuarray  # type: ignore
from pycuda.compiler import SourceModule  # type: ignore
from pycuda.driver import Function  # type: ignore

from pydtnn.activations.leaky_relu import LeakyRelu
from pydtnn.backends.pycuda.activations.activation import ActivationPycuda
from pydtnn.backends.pycuda.utils.tensor_array import TensorArray
from pydtnn.tracers.events import (PYDTNN_OPS_EVENT, PYDTNN_OPS_EVENTS,
                                   PYDTNN_OPS_EVENT_enum)
from pydtnn.utils.constants import DTYPE2CTYPE, ArrayShape
from pydtnn.utils.performance_models import col2im_time, im2col_time

__all__=(
    "LeakyReluPycuda",
)

logger = logging.getLogger(__name__)


class LeakyReluPycuda(LeakyRelu[TensorArray], ActivationPycuda):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # The following attributes will be initialized later.
        self.mask: TensorArray = None  # type: ignore
        self.y: TensorArray = None  # type: ignore

    def _model_init(self, prev_shape: ArrayShape, x: TensorArray) -> None:
        super()._model_init(prev_shape, x)

        y_gpu = gpuarray.zeros(x.shape, self.model.dtype)
        self.y = TensorArray(y_gpu, self.model.tensor_format, self.model.cudnn_dtype)

        mask_gpu = gpuarray.zeros((self.model.batch_size, *self.prev_shape), self.model.dtype)
        self.mask = TensorArray(mask_gpu, self.model.tensor_format, self.model.cudnn_dtype)

        self.memory_used += self.y.nbytes + self.mask.nbytes

        self.defines_replaces = {"\"TYPE\"": DTYPE2CTYPE[self.model.dtype]}
        self.cuda_fwd_func = self._fwd_kernel()
        self.cuda_bwd_func = self._bwd_kernel()

        self.total_num_threads = np.int32(math.prod(self.grid) * math.prod(self.block))

        self.initialize_relu_2d_gpu(prev_shape)

    def initialize_relu_2d_gpu(self, prev_shape: ArrayShape) -> None:
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

        self.fwd_time = \
            im2col_time(m=self.ci, n=n, cpu_speed=self.model.cpu_speed,
                        memory_bw=self.model.memory_bw, dtype=self.model.dtype)  # type: ignore (it's fine)
        self.bwd_time = \
            col2im_time(m=self.ci, n=n, cpu_speed=self.model.cpu_speed,
                        memory_bw=self.model.memory_bw, dtype=self.model.dtype)  # type: ignore (it's fine)

    def forward(self, x: TensorArray) -> TensorArray:
        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, self.id * PYDTNN_OPS_EVENTS + PYDTNN_OPS_EVENT_enum.FORWARD_CUDNN)

        n = np.int32(math.prod(x.shape))

        self.cuda_fwd_func(x.ary, self.mask.ary, self.max.ary,
                           np.float32(self.negative_slope), self.total_num_threads, n,
                           grid=self.grid, block=self.block, stream=self.model.stream)

        self.y: TensorArray = self.mask

        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, 0)

        return self.y

    def backward(self, dy: TensorArray) -> TensorArray:
        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, self.id * PYDTNN_OPS_EVENTS + PYDTNN_OPS_EVENT_enum.BACKWARD_CUDNN_DX)

        n = np.int32(math.prod(dy.shape))

        self.cuda_bwd_func(self.dx.ary, dy.ary, self.mask.ary,
                           self.total_num_threads, n,
                           grid=self.grid, block=self.block,
                           stream=self.model.stream)

        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, 0)
        return self.dx
