"""PyCUDA implementation of the Adaptive Average Pooling 2D layer."""

import logging
import math

import numpy as np

from pydtnn.backends.pycuda.layers.abstract.layer import LayerPycuda
from pydtnn.layers.adaptive_average_pool_2d import AdaptiveAveragePool2D
from pydtnn.tracers.abstract.events import (PYDTNN_EVENT_FINISHED, PYDTNN_OPS_EVENT,
                                            PYDTNN_OPS_EVENTS, OpsEventEnum)
from pydtnn.utils.constants import ArrayShape
from pydtnn.utils.performance_models import col2im_time, im2col_time
from pydtnn.utils.tensor_array import FormatType, MemoryType, TensorArray

__all__ = ("AdaptiveAveragePool2DPycuda",)

logger = logging.getLogger(__name__)


# Import from AbstractPool2DLayerPycuda


class AdaptiveAveragePool2DPycuda(AdaptiveAveragePool2D[TensorArray], LayerPycuda):
    """PyCUDA-accelerated Adaptive Average Pooling 2D layer."""

    def _model_init(self, prev_shape: ArrayShape, x: TensorArray) -> None:
        """Initialize model parameters and CUDA kernels."""
        super()._model_init(prev_shape, x)

        self.cuda_fwd_func = self._fwd_kernel()
        self.cuda_bwd_func = self._bwd_kernel()

        self.initialize_pool_2d_gpu(prev_shape, x)

    def initialize_pool_2d_gpu(self, prev_shape: ArrayShape, x: TensorArray) -> None:
        """Allocate GPU memory and initialize performance metrics for the pooling layer."""
        self.ci, self.hi, self.wi = self.model.decode_shape(prev_shape)
        self.shape = self.model.encode_shape((self.co, self.ho, self.wo))
        pooling_shape = self.model.encode_shape((self.co, self.ho, self.wo))

        self.y = TensorArray.zeros(
            (self.model.batch_size, *pooling_shape),
            self.model.dtype,
            tensor_format=self.model.tensor_format,
            mem_type=MemoryType.GLOBAL if self.model.use_gpudirect else MemoryType.DEVICE,
            fmt_type=FormatType.TENSOR
        )

        # Derivative dx
        self.dx = TensorArray.zeros(
            self.x.ary.shape, self.model.dtype,
            tensor_format=self.model.tensor_format,
            mem_type=MemoryType.GLOBAL if self.model.use_gpudirect else MemoryType.DEVICE,
            fmt_type=FormatType.TENSOR
        )

        self.memory_used += self.y.nbytes + self.dx.nbytes

        self.fwd_time = im2col_time(
            m=self.co,
            n=(self.model.batch_size * self.ho * self.wo * self.ci),
            cpu_speed=self.model.cpu_speed,
            memory_bw=self.model.memory_bw,
            dtype=self.model.dtype,
        )
        self.bwd_time = col2im_time(
            m=self.co,
            n=(self.model.batch_size * self.ho * self.wo * self.ci),
            cpu_speed=self.model.cpu_speed,
            memory_bw=self.model.memory_bw,
            dtype=self.model.dtype,
        )

    def forward(self, x: TensorArray) -> TensorArray:
        """Perform the forward pass of the adaptive average pooling operation."""
        self.model.tracer.emit_event(
            PYDTNN_OPS_EVENT, self.id * PYDTNN_OPS_EVENTS + OpsEventEnum.FORWARD_CUDNN
        )

        if self.pooling_not_needed:
            self.y = x
        else:
            n, c, h, w = self.model.decode_shape(x.shape)
            num_elements = np.int32(math.prod((n, c, self.ho, self.wo)))

            # NOTE: Instead of a number, PyCuda's driver expects "numpy.number"
            self.cuda_fwd_func(
                x.ary,
                self.y.ary,
                np.int32(n),
                np.int32(c),
                np.int32(h),
                np.int32(w),
                np.int32(self.ho),
                np.int32(self.wo),
                num_elements,
                grid=self.grid,
                block=self.block,
                stream=self.model.pycuda_stream,
            )

        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, PYDTNN_EVENT_FINISHED)
        return self.y

    def backward(self, dy: TensorArray) -> TensorArray:
        """Perform the backward pass of the adaptive average pooling operation."""
        self.model.tracer.emit_event(
            PYDTNN_OPS_EVENT, self.id * PYDTNN_OPS_EVENTS + OpsEventEnum.BACKWARD_CUDNN_DX
        )
        n, c, h, w = self.model.decode_shape(dy.shape)
        self.dx.fill(0)

        self.cuda_bwd_func(
            self.dx.ary,
            dy.ary,
            np.int32(n),
            np.int32(c),
            np.int32(h),
            np.int32(w),
            np.int32(self.hi),
            np.int32(self.wi),
            grid=self.grid,
            block=self.block,
            stream=self.model.pycuda_stream,
        )

        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, PYDTNN_EVENT_FINISHED)
        return self.dx
