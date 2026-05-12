"""
PyCUDA implementation of a 2D pointwise convolution layer.
"""
import logging
from typing import Any, override

import numpy as np
from pycuda import gpuarray  # type: ignore
from pycuda.driver import Function  # type: ignore

from pydtnn.backends.pycuda.layers.abstract.conv_2d import AbstractConv2DPycuda
from pydtnn.backends.pycuda.utils.tensor_array import TensorArray
from pydtnn.tracers.events import PYDTNN_EVENT_FINISHED, PYDTNN_OPS_EVENT, PYDTNN_OPS_EVENTS, PYDTNN_OPS_EVENT_enum
from pydtnn.utils.constants import ArrayShape
from pydtnn.utils.tensor import TensorFormat, format_transpose

__all__ = ("Conv2DPointwisePycuda",)

logger = logging.getLogger(__name__)


class Conv2DPointwisePycuda(AbstractConv2DPycuda):
    """
    PyCUDA-accelerated 2D pointwise convolution layer (1x1 kernel).
    """
    def _initializing_special_parameters(self):
        """
        Initializes kernel dimensions and weight shapes for pointwise convolution.
        """
        self.kh = self.kw = 1
        # Setting weights
        match self.model.tensor_format:
            case TensorFormat.NCHW:
                self.weights_shape = (self.co, self.ci)
            case TensorFormat.NHWC:
                self.weights_shape = (self.co, self.ci)
            case _:
                raise NotImplementedError(f"{self.model.tensor_format} format not implemented.")

    def _model_init(self, prev_shape: ArrayShape, x: TensorArray) -> None:
        """
        Initializes model buffers, kernels, and memory allocations for the layer.
        """
        super()._model_init(prev_shape, x)
        self.bias_sum_bwd: Function = None

        match self.model.tensor_format:
            case TensorFormat.NCHW:
                # self.bias_sum_bwd = self.cuda_sum_bias_axis_023()
                self.bias_sum_bwd = self._get_kernel(code_file_name="conv2d", func_name="cuda_sum_bias_axis_023")
            case TensorFormat.NHWC:
                # self.bias_sum_bwd = self.cuda_sum_bias_axis_012()
                self.bias_sum_bwd = self._get_kernel(code_file_name="conv2d", func_name="cuda_sum_bias_axis_012")
            case _:
                raise NotImplementedError(f"conv_2d_gpu_depthwise is not implemented for {self.model.tensor_format} format.")

        self.total_num_threads = np.int32(np.prod(self.grid) * np.prod(self.block))

        y_gpu = gpuarray.to_gpu(np.zeros(shape=(self.model.batch_size, *self.shape), dtype=self.model.dtype))
        self.y = TensorArray(y_gpu, self.model.tensor_format, self.model.cudnn_dtype)
        self.memory_used += self.y.nbytes

        dx_gpu = gpuarray.zeros((self.model.batch_size, *self.shape), self.model.dtype)
        self.dx = TensorArray(dx_gpu, self.model.tensor_format, self.model.cudnn_dtype)

        self.memory_used += self.dx.nbytes

        self.forward = self._forward_pointwise
        self.backward = self._backward_pointwise
        self.fwd_func: Function = self._fwd_kernel
        self.bwd_func: Function = self._fwd_kernel
        self.bias_sum_fwd: Function = self._get_kernel(func_name="cuda_bias_sum_fwd_pointwise_conv")

    def _forward_pointwise(self, x: TensorArray) -> TensorArray:
        """
        Performs the forward pass of the pointwise convolution.
        """

        self.x = x
        self.y.fill(0)

        n, c, h, w = self.model.decode_shape(x.shape)  # type: ignore (it's okay)

        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, self.id * PYDTNN_OPS_EVENTS + PYDTNN_OPS_EVENT_enum.FORWARD_CUDNN)
        self.fwd_func(
            x.ary,
            self.weights.ary,
            self.y.ary,
            np.int32(n),
            np.int32(c),
            np.int32(h),
            np.int32(w),
            np.int32(self.co),
            self.total_num_threads,
            grid=self.grid,
            block=self.block,
            stream=self.model.stream,
        )

        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, PYDTNN_EVENT_FINISHED)

        if self.use_bias:
            self.biases: TensorArray
            self.model.tracer.emit_event(PYDTNN_OPS_EVENT, self.id * PYDTNN_OPS_EVENTS + PYDTNN_OPS_EVENT_enum.FORWARD_CUDNN_SUM_BIASES)
            self.bias_sum_fwd(
                x.ary, self.biases.ary, np.int32(n), np.int32(c), np.int32(h), np.int32(w), np.int32(n * c * h * w), self.total_num_threads, grid=self.grid, block=self.block, stream=self.model.stream
            )
            self.model.tracer.emit_event(PYDTNN_OPS_EVENT, PYDTNN_EVENT_FINISHED)

        return self.y

    def _backward_pointwise(self, dy: TensorArray) -> TensorArray:
        """
        Performs the backward pass of the pointwise convolution.
        """
        n, c, h, w = self.model.decode_shape(dy.shape)  # type: ignore (it's okay)
        self.dx.fill(0)

        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, self.id * PYDTNN_OPS_EVENTS + PYDTNN_OPS_EVENT_enum.BACKWARD_CUDNN_DX)
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
            self.model.tracer.emit_event(PYDTNN_OPS_EVENT, self.id * PYDTNN_OPS_EVENTS + PYDTNN_OPS_EVENT_enum.BACKWARD_CUDNN_DB)
            self.bias_sum_bwd(dy.ary, self.db.ary, np.int32(c), np.int32(h), np.int32(w), np.int32(n * c * h * w), self.total_num_threads, grid=self.grid, block=self.block, stream=self.model.stream)
            self.model.tracer.emit_event(PYDTNN_OPS_EVENT, PYDTNN_EVENT_FINISHED)

        return self.dx

    @override
    def _export_weights_dw(self, key: str) -> Any:
        """
        Exports weights or gradients to CPU, handling format transposition if necessary.
        """
        value = getattr(self, key)

        match self.model.tensor_format:
            case TensorFormat.NHWC:
                # NHWC's src: self.ci, self.co
                # NCHW's dst: self.co, self.ci
                gpu_ary = value
                cpu_ary = gpu_ary.get()
                return np.asarray(format_transpose(cpu_ary, "IO", "OI"), dtype=np.float64, order="C", copy=True)
            case TensorFormat.NCHW:
                # NHWC's src: self.ci, self.co
                # NCHW's dst: self.co, self.ci
                gpu_ary = value
                cpu_ary = gpu_ary.get()
                return np.asarray(cpu_ary, dtype=np.float64, order="C", copy=True)
            case _:
                return super()._export_prop(key)

    @override
    def _import_weights_dw(self, key: str, value: Any) -> None:
        """
        Imports weights or gradients from CPU, handling format transposition if necessary.
        """
        attribute = getattr(self, key)
        match self.model.tensor_format:
            case TensorFormat.NHWC:
                # NHWC's src: self.ci, self.co
                # NCHW's dst: self.co, self.ci
                cpu_ary = format_transpose(value, "OI", "IO")
                attribute.set(cpu_ary)
                return
            case TensorFormat.NCHW:
                cpu_ary = value
                attribute.set(cpu_ary)
                return
            case _:
                return super()._export_prop(key)