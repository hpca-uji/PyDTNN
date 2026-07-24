"""PyCUDA implementation of Depthwise 2D Convolution layer."""

import logging
from typing import Any, override

import numpy as np
from pycuda import gpuarray  # pyright: ignore[reportAttributeAccessIssue]
from pycuda.driver import Function

from pydtnn.backends.pycuda.layers.abstract.conv_2d import AbstractConv2DPycuda
from pydtnn.backends.pycuda.utils.tensor_array import TensorArray
from pydtnn.tracers.events import (PYDTNN_EVENT_FINISHED, PYDTNN_OPS_EVENT,
                                   PYDTNN_OPS_EVENTS, OpsEventEnum)
from pydtnn.utils.constants import ArrayShape
from pydtnn.utils.tensor import TensorFormat

__all__ = ("Conv2DDepthwisePycuda",)

logger = logging.getLogger(__name__)


class Conv2DDepthwisePycuda(AbstractConv2DPycuda):
    """Depthwise 2D Convolution layer implementation for PyCUDA backend."""

    def _initializing_special_parameters(self) -> None:
        """Initialize layer-specific parameters and weight shapes."""
        # Setting other parameters
        self.co = self.ci
        # Setting weights
        self.weights_shape = (1, self.ci, *self.filter_shape)

    def _model_init(self, prev_shape: ArrayShape, x: TensorArray) -> None:
        """Initialize model buffers and select CUDA kernels based on tensor format."""
        super()._model_init(prev_shape, x)
        self.bias_sum_bwd: Function = None

        match self.model.tensor_format:
            case TensorFormat.NCHW:
                # self.bias_sum_bwd = self.cuda_sum_bias_axis_023()
                self.bias_sum_bwd = self._get_kernel(
                    code_file_name="conv2d", func_name="cuda_sum_bias_axis_023"
                )
                self.forward = self._forward_depthwise_nchw
                self.backward = self._backward_depthwise_nchw
            case TensorFormat.NHWC:
                # self.bias_sum_bwd = self.cuda_sum_bias_axis_012()
                self.bias_sum_bwd = self._get_kernel(
                    code_file_name="conv2d", func_name="cuda_sum_bias_axis_012"
                )
                self.forward = self._forward_depthwise_nhwc
                self.backward = self._backward_depthwise_nhwc
            case _:
                # TODO: self devolvía la versión con el número
                raise NotImplementedError(
                    f"{self.name} is not implemented for {self.model.tensor_format} format."
                )

        self.total_num_threads = np.int32(np.prod(self.grid) * np.prod(self.block))

        y_gpu = gpuarray.zeros((self.model.batch_size, *self.shape), self.model.dtype)
        self.y = TensorArray(y_gpu, self.model.tensor_format, self.model.cudnn_dtype)
        self.memory_used += self.y.nbytes

        dx_gpu = gpuarray.zeros((self.model.batch_size, *self.shape), self.model.dtype)
        self.dx = TensorArray(dx_gpu, self.model.tensor_format, self.model.cudnn_dtype)
        self.memory_used += self.dx.nbytes

        self.fwd_func: Function = self._fwd_kernel()
        self.bwd_func: Function = self._bwd_kernel()
        self.bias_sum_fwd: Function = self._get_kernel(func_name="cuda_bias_sum_fwd_depthwise_conv")

    def _forward_depthwise_nchw(self, x: TensorArray) -> TensorArray:
        """Execute forward pass for NCHW format."""
        self.x = x
        self.y.fill(0)

        n, c, h, w = x.shape

        self.model.tracer.emit_event(
            PYDTNN_OPS_EVENT, self.id * PYDTNN_OPS_EVENTS + OpsEventEnum.FORWARD_CUDNN
        )
        self.fwd_func(
            x.ary,
            self.weights.ary,
            self.y.ary,
            np.int32(self.hpadding),
            np.int32(self.wpadding),
            np.int32(self.hstride),
            np.int32(self.wstride),
            np.int32(self.hdilation),
            np.int32(self.wdilation),
            np.int32(n),
            np.int32(c),
            np.int32(h),
            np.int32(w),
            np.int32(self.kh),
            np.int32(self.kw),
            np.int32(self.ho),
            np.int32(self.wo),
            self.total_num_threads,
            grid=self.grid,
            block=self.block,
            stream=self.model.stream,
        )

        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, PYDTNN_EVENT_FINISHED)

        if self.use_bias:
            self.biases: TensorArray
            self.model.tracer.emit_event(
                PYDTNN_OPS_EVENT,
                self.id * PYDTNN_OPS_EVENTS + OpsEventEnum.FORWARD_CUDNN_SUM_BIASES,
            )
            self.bias_sum_fwd(
                x.ary,
                self.biases.ary,
                np.int32(n),
                np.int32(c),
                np.int32(h),
                np.int32(w),
                np.int32(n * h * w * c),
                self.total_num_threads,
                grid=self.grid,
                block=self.block,
                stream=self.model.stream,
            )
            self.model.tracer.emit_event(PYDTNN_OPS_EVENT, PYDTNN_EVENT_FINISHED)

        return self.y

    def _forward_depthwise_nhwc(self, x: TensorArray) -> TensorArray:
        """Execute forward pass for NHWC format."""
        self.x = x
        n, h, w, c = x.shape
        self.y.fill(0)

        self.model.tracer.emit_event(
            PYDTNN_OPS_EVENT, self.id * PYDTNN_OPS_EVENTS + OpsEventEnum.FORWARD_CUDNN
        )
        self.fwd_func(
            x.ary,
            self.weights.ary,
            self.y.ary,
            np.int32(self.hpadding),
            np.int32(self.wpadding),
            np.int32(self.hstride),
            np.int32(self.wstride),
            np.int32(self.hdilation),
            np.int32(self.wdilation),
            np.int32(n),
            np.int32(c),
            np.int32(h),
            np.int32(w),
            np.int32(self.kh),
            np.int32(self.kw),
            np.int32(self.ho),
            np.int32(self.wo),
            self.total_num_threads,
            grid=self.grid,
            block=self.block,
            stream=self.model.stream,
        )

        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, PYDTNN_EVENT_FINISHED)

        if self.use_bias:
            self.biases: TensorArray
            self.model.tracer.emit_event(
                PYDTNN_OPS_EVENT,
                self.id * PYDTNN_OPS_EVENTS + OpsEventEnum.FORWARD_CUDNN_SUM_BIASES,
            )
            self.bias_sum_fwd(
                x.ary,
                self.biases.ary,
                np.int32(n),
                np.int32(c),
                np.int32(h),
                np.int32(w),
                np.int32(n * h * w * c),
                self.total_num_threads,
                grid=self.grid,
                block=self.block,
                stream=self.model.stream,
            )
            self.model.tracer.emit_event(PYDTNN_OPS_EVENT, PYDTNN_EVENT_FINISHED)

        return self.y

    def _backward_depthwise_nchw(self, dy: TensorArray) -> TensorArray:
        """Execute backward pass for NCHW format."""

        n, c, h, w = dy.shape
        self.dx.fill(0)

        self.model.tracer.emit_event(
            PYDTNN_OPS_EVENT, self.id * PYDTNN_OPS_EVENTS + OpsEventEnum.BACKWARD_CUDNN_DX
        )
        self.fwd_func(
            dy.ary,
            self.x.ary,
            self.weights.ary,
            self.dx.ary,
            self.dw.ary,
            np.int32(self.hpadding),
            np.int32(self.wpadding),
            np.int32(self.hstride),
            np.int32(self.wstride),
            np.int32(self.hdilation),
            np.int32(self.wdilation),
            np.int32(n),
            np.int32(c),
            np.int32(h),
            np.int32(w),
            np.int32(self.kh),
            np.int32(self.kw),
            np.int32(self.ho),
            np.int32(self.wo),
            self.total_num_threads,
            grid=self.grid,
            block=self.block,
            stream=self.model.stream,
        )
        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, PYDTNN_EVENT_FINISHED)

        if self.use_bias:
            self.biases: TensorArray
            self.model.tracer.emit_event(
                PYDTNN_OPS_EVENT,
                self.id * PYDTNN_OPS_EVENTS + OpsEventEnum.BACKWARD_CUDNN_DB,
            )
            self.bias_sum_bwd(
                dy.ary,
                self.db.ary,
                np.int32(c),
                np.int32(h),
                np.int32(w),
                np.int32(n * c * h * w),
                self.total_num_threads,
                grid=self.grid,
                block=self.block,
                stream=self.model.stream,
            )
            self.model.tracer.emit_event(PYDTNN_OPS_EVENT, PYDTNN_EVENT_FINISHED)

        return self.dx

    def _backward_depthwise_nhwc(self, dy: TensorArray) -> TensorArray:
        """Execute backward pass for NHWC format."""
        n, h, w, c = dy.shape
        self.dx.fill(0)

        self.model.tracer.emit_event(
            PYDTNN_OPS_EVENT, self.id * PYDTNN_OPS_EVENTS + OpsEventEnum.BACKWARD_CUDNN_DX
        )
        self.fwd_func(
            dy.ary,
            self.x.ary,
            self.weights.ary,
            self.dx.ary,
            self.dw.ary,
            np.int32(self.hpadding),
            np.int32(self.wpadding),
            np.int32(self.hstride),
            np.int32(self.wstride),
            np.int32(self.hdilation),
            np.int32(self.wdilation),
            np.int32(n),
            np.int32(c),
            np.int32(h),
            np.int32(w),
            np.int32(self.kh),
            np.int32(self.kw),
            np.int32(self.ho),
            np.int32(self.wo),
            self.total_num_threads,
            grid=self.grid,
            block=self.block,
            stream=self.model.stream,
        )
        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, PYDTNN_EVENT_FINISHED)

        if self.use_bias:
            self.biases: TensorArray
            self.model.tracer.emit_event(
                PYDTNN_OPS_EVENT,
                self.id * PYDTNN_OPS_EVENTS + OpsEventEnum.BACKWARD_CUDNN_DB,
            )
            self.bias_sum_bwd(
                dy.ary,
                self.db.ary,
                np.int32(c),
                np.int32(n * h * w * c),
                self.total_num_threads,
                grid=self.grid,
                block=self.block,
                stream=self.model.stream,
            )
            self.model.tracer.emit_event(PYDTNN_OPS_EVENT, PYDTNN_EVENT_FINISHED)

        return self.dx

    @override
    def _export_weights_dw(self, key: str) -> Any:
        """Export weights from GPU to CPU."""
        value = getattr(self, key)
        gpu_ary = value
        cpu_ary = gpu_ary.get()
        return np.asarray(cpu_ary, dtype=np.float64, order="C", copy=True)

    @override
    def _import_weights_dw(self, key: str, value: Any) -> None:
        """Import weights from CPU to GPU."""
        attribute = getattr(self, key)
        attribute.set(value)
