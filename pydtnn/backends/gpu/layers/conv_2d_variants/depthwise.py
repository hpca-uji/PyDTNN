import numpy as np

from pydtnn.tracers.events import PYDTNN_OPS_EVENT, PYDTNN_OPS_EVENTS, PYDTNN_EVENT_FINISHED, PYDTNN_OPS_EVENT_enum
from pydtnn.backends.gpu.layers.conv_2d import Conv2DGPU, MACROS_NCHW, MACROS_NHWC
from pydtnn.backends.gpu.utils.tensor_gpu import TensorGPU
from pydtnn.utils.constants import ArrayShape, DTYPE2CTYPE

from pydtnn.utils.tensor import TensorFormat
from typing import Any, override

import pycuda.gpuarray as gpuarray  #type: ignore
from pycuda.compiler import SourceModule  #type: ignore
from pycuda.driver import Function  #type: ignore

class Conv2DDepthwiseGPU(Conv2DGPU):

    def _initializing_special_parameters(self):
        # Setting other parameters
        self.co = self.ci
        # Setting weights
        self.weights_shape = (1, self.ci, *self.filter_shape)

    def initialize(self, prev_shape: ArrayShape, x: TensorGPU) -> None:
        super().initialize(prev_shape, x)

        func_name: str = ""
        macros: str = ""
        self.bias_sum_bwd: Function = None

        match self.model.tensor_format:
            case TensorFormat.NCHW:
                func_name = "cuda_depthwise_conv_2d_{fwd_bwd}_nchw"
                macros = MACROS_NCHW
                self.bias_sum_bwd = self.cuda_sum_bias_axis_023()
                self.forward = self._forward_depthwise_nchw
                self.backward = self._backward_depthwise_nchw
            case TensorFormat.NHWC:
                func_name = "cuda_depthwise_conv_2d_{fwd_bwd}_nhwc"
                macros = MACROS_NHWC
                self.bias_sum_bwd = self.cuda_sum_bias_axis_012()
                self.forward = self._forward_depthwise_nhwc
                self.backward = self._backward_depthwise_nhwc
            case _:
                # TODO: self devolvía la versión con el número
                raise NotImplementedError(f"\"{self.name}\" is not implemented for \"{self.model.tensor_format}\" format.")

        self.total_num_threads = np.prod(self.grid, dtype=np.int32) * np.prod(self.block, dtype=np.int32)

        y_gpu = gpuarray.zeros((self.model.batch_size, *self.shape), self.model.dtype)
        self.y = TensorGPU(y_gpu, self.model.tensor_format, self.model.cudnn_dtype)

        dx_gpu = gpuarray.zeros((self.model.batch_size, *self.shape), self.model.dtype)
        self.dx = TensorGPU(dx_gpu, self.model.tensor_format, self.model.cudnn_dtype)

        self.fwd_func: Function = self.cuda_depthwise_conv_2d_fwd(func_name.format(fwd_bwd="fwd"), macros)
        self.bwd_func: Function = self.cuda_depthwise_conv_2d_bwd(func_name.format(fwd_bwd="bwd"), macros)
        self.bias_sum_fwd: Function = self.cuda_bias_sum_fwd_depthwise_conv()
    # ----

    def _forward_depthwise_nchw(self, x: TensorGPU) -> TensorGPU:
        self.x = x
        self.y.fill(0)

        n, c, h, w = x.shape

        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, self.id * PYDTNN_OPS_EVENTS + PYDTNN_OPS_EVENT_enum.FORWARD_CUDNN)
        self.fwd_func(x.ary, self.weights.ary, self.y.ary,
                      np.int32(self.vpadding), np.int32(self.hpadding),
                      np.int32(self.vstride), np.int32(self.hstride),
                      np.int32(self.vdilation), np.int32(self.hdilation),
                      np.int32(n), np.int32(c), np.int32(h), np.int32(w),
                      np.int32(self.kh), np.int32(self.kw), np.int32(self.ho), np.int32(self.wo),
                      self.total_num_threads, grid=self.grid, block=self.block,
                      stream=self.model.stream)

        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, PYDTNN_EVENT_FINISHED)

        if self.use_bias:
            self.biases: TensorGPU
            self.model.tracer.emit_event(PYDTNN_OPS_EVENT, self.id * PYDTNN_OPS_EVENTS + PYDTNN_OPS_EVENT_enum.FORWARD_CUDNN_SUM_BIASES)
            self.bias_sum_fwd(x.ary, self.biases.ary,
                              np.int32(n), np.int32(c), np.int32(h), np.int32(w),
                              np.int32(n * h * w * c),
                              self.total_num_threads,
                              grid=self.grid, block=self.block,
                              stream=self.model.stream)
            self.model.tracer.emit_event(PYDTNN_OPS_EVENT, PYDTNN_EVENT_FINISHED)

        return self.y
    # ----

    def _forward_depthwise_nhwc(self, x: TensorGPU) -> TensorGPU:
        self.x = x
        n, h, w, c = x.shape
        self.y.fill(0)

        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, self.id * PYDTNN_OPS_EVENTS + PYDTNN_OPS_EVENT_enum.FORWARD_CUDNN)
        self.fwd_func(x.ary, self.weights.ary, self.y.ary,
                      np.int32(self.vpadding), np.int32(self.hpadding),
                      np.int32(self.vstride), np.int32(self.hstride),
                      np.int32(self.vdilation), np.int32(self.hdilation),
                      np.int32(n), np.int32(c), np.int32(h), np.int32(w),
                      np.int32(self.kh), np.int32(self.kw), np.int32(self.ho), np.int32(self.wo),
                      self.total_num_threads, grid=self.grid, block=self.block,
                      stream=self.model.stream)

        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, PYDTNN_EVENT_FINISHED)

        if self.use_bias:
            self.biases: TensorGPU
            self.model.tracer.emit_event(PYDTNN_OPS_EVENT, self.id * PYDTNN_OPS_EVENTS + PYDTNN_OPS_EVENT_enum.FORWARD_CUDNN_SUM_BIASES)
            self.bias_sum_fwd(x.ary, self.biases.ary,
                              np.int32(n), np.int32(c), np.int32(h), np.int32(w),
                              np.int32(n * h * w * c),
                              self.total_num_threads,
                              grid=self.grid, block=self.block,
                              stream=self.model.stream)
            self.model.tracer.emit_event(PYDTNN_OPS_EVENT, PYDTNN_EVENT_FINISHED)

        return self.y
    # ----

    def _backward_depthwise_nchw(self, dy: TensorGPU) -> TensorGPU:

        n, c, h, w = dy.shape
        self.dx.fill(0)

        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, self.id * PYDTNN_OPS_EVENTS + PYDTNN_OPS_EVENT_enum.BACKWARD_CUDNN_DX)
        self.fwd_func(dy.ary, self.x.ary, self.weights.ary,
                      self.dx.ary, self.dw.ary,
                      np.int32(self.vpadding), np.int32(self.hpadding),
                      np.int32(self.vstride), np.int32(self.hstride),
                      np.int32(self.vdilation), np.int32(self.hdilation),
                      np.int32(n), np.int32(c), np.int32(h), np.int32(w),
                      np.int32(self.kh), np.int32(self.kw), np.int32(self.ho), np.int32(self.wo),
                      self.total_num_threads,
                      grid=self.grid, block=self.block, stream=self.model.stream)
        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, PYDTNN_EVENT_FINISHED)

        if self.use_bias:
            self.biases: TensorGPU
            self.model.tracer.emit_event(PYDTNN_OPS_EVENT, self.id * PYDTNN_OPS_EVENTS + PYDTNN_OPS_EVENT_enum.BACKWARD_CUDNN_DB)
            self.bias_sum_bwd(dy.ary, self.db.ary,
                              np.int32(c), np.int32(h), np.int32(w),
                              np.int32(n * c * h * w), self.total_num_threads,
                              grid=self.grid, block=self.block, stream=self.model.stream)
            self.model.tracer.emit_event(PYDTNN_OPS_EVENT, PYDTNN_EVENT_FINISHED)

        return self.dx
    # -----

    def _backward_depthwise_nhwc(self, dy: TensorGPU) -> TensorGPU:
        n, h, w, c = dy.shape
        self.dx.fill(0)

        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, self.id * PYDTNN_OPS_EVENTS + PYDTNN_OPS_EVENT_enum.BACKWARD_CUDNN_DX)
        self.fwd_func(dy.ary, self.x.ary, self.weights.ary,
                      self.dx.ary, self.dw.ary,
                      np.int32(self.vpadding), np.int32(self.hpadding),
                      np.int32(self.vstride), np.int32(self.hstride),
                      np.int32(self.vdilation), np.int32(self.hdilation),
                      np.int32(n), np.int32(c), np.int32(h), np.int32(w),
                      np.int32(self.kh), np.int32(self.kw), np.int32(self.ho), np.int32(self.wo),
                      self.total_num_threads, grid=self.grid, block=self.block,
                      stream=self.model.stream)
        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, PYDTNN_EVENT_FINISHED)

        if self.use_bias:
            self.biases: TensorGPU
            self.model.tracer.emit_event(PYDTNN_OPS_EVENT, self.id * PYDTNN_OPS_EVENTS + PYDTNN_OPS_EVENT_enum.BACKWARD_CUDNN_DB)
            self.bias_sum_bwd(dy.ary, self.db.ary,
                              np.int32(c), np.int32(n * h * w * c),
                              self.total_num_threads,
                              grid=self.grid, block=self.block, stream=self.model.stream)
            self.model.tracer.emit_event(PYDTNN_OPS_EVENT, PYDTNN_EVENT_FINISHED)

        return self.dx
    # -----

    @override
    def _export_weights_dw(self, key: str) -> Any:
        value = getattr(self, key)

        match self.model.tensor_format:
            case TensorFormat.NHWC:
                gpu_ary = value.ary
                cpu_ary = np.squeeze(gpu_ary.get(), axis=0)
                return np.asarray(cpu_ary, dtype=np.float64, order="C", copy=True)
            case TensorFormat.NCHW:
                gpu_ary = value.ary
                cpu_ary = np.squeeze(gpu_ary.get(), axis=0)
                return np.asarray(cpu_ary, dtype=np.float64, order="C", copy=True)
            case default:
                return super()._export_prop(key)
    # ---

    @override
    def _import_weights_dw(self, key: str, value: Any) -> None:
        attribute = getattr(self, key)
        match self.model.tensor_format:
            case TensorFormat.NHWC:
                cpu_ary = np.asarray(value, dtype=self.model.dtype, order="C", copy=None)
                cpu_ary = np.expand_dims(cpu_ary, axis=0)
                attribute.ary.set(cpu_ary)
                return
            case TensorFormat.NCHW:
                gpu_ary = attribute.ary
                cpu_ary = np.asarray(np.expand_dims(value, axis=0), dtype=self.model.dtype, order="C", copy=None)
                gpu_ary.set(cpu_ary)
                return 
            case default:
                return super()._import_prop(key, value)
    # ---


#########################################################################################################
## CUDA CODE ##
###############

    def cuda_depthwise_conv_2d_fwd(self, _func_name: str, _macros: str) -> Function:

        code = \
"""
{macros}
__global__ void {func_name}({T}* x, {T}* k, {T}* res,
                            int vpadding, int hpadding,
                            int vstride, int hstride,
                            int vdilation, int hdilation,
                            int n, int c, int h, int w,
                            int kh, int kw, int ho, int wo,
                            int num_workers)
{{
    int idx, cc, hi, wi, yy, xx, nn, x_x, x_y;
    int N = n * c * ho * wo;
    {T} val_k, val_x;

    for(idx = blockIdx.x * blockDim.x + threadIdx.x; idx < N; idx += num_workers)
    {{
        cc = INDEX_C(idx, c, ho, wo);
        xx = INDEX_H(idx, c, ho, wo);
        yy = INDEX_W(idx, c, ho, wo);

        for (hi = 0; hi < kh; hi++)
        {{
            for (wi = 0; wi < kw; wi++)
            {{
                x_x = vstride * xx + vdilation * hi - vpadding;
                x_y = hstride * yy + hdilation * wi - hpadding;
                if ((0 <= x_x) && (x_x < h) && (0 <= x_y) && (x_y < w))
                {{
                    val_k = *(SHIFT_POINTER(k, c, h, w, 0, cc, hi, wi));
                    val_x = *(SHIFT_POINTER(x, c, h, w, nn, cc, x_x, x_y));
                    *(SHIFT_POINTER(res, c, h, w, nn, cc, xx, yy)) += ({T}) (val_k * val_x);
                }}
            }}
        }}
    }}
}}
"""
        _t = DTYPE2CTYPE[self.model.dtype]  # variable Type

        code = code.format(macros=_macros,
                           func_name=_func_name,
                           T=_t
                           )
        module = SourceModule(code).get_function(_func_name)

        return module
    # ---

    def cuda_depthwise_conv_2d_bwd(self, _func_name: str, _macros: str) -> Function:

        code = \
"""
{macros}
__global__ void {func_name}({T}* dy, {T}* x, {T}* k,
                            {T}* dx, {T}* dw,
                            int vpadding, int hpadding,
                            int vstride, int hstride,
                            int vdilation, int hdilation,
                            int n, int c, int h, int w,
                            int kh, int kw, int ho, int wo,
                            int num_workers)
{{
    int idx, cc, khi, kwi, yy, xx, nn, x_x, x_y;
    {T} val_k, val_dy, val_x;
    int N = n * c * ho * wo;

    for(idx = blockIdx.x * blockDim.x + threadIdx.x; idx < N; idx += num_workers)
    {{
        cc = INDEX_C(idx, c, ho, wo);
        xx = INDEX_H(idx, c, ho, wo);
        yy = INDEX_W(idx, c, ho, wo);

        val_dy = ({T}) *(SHIFT_POINTER(dy, c, h, w, nn, cc, xx, yy));
        for (khi = 0; khi < kh; khi++)
        {{
            for (kwi = 0; kwi < kw; kwi++)
            {{
                x_x = vstride * xx + vdilation * khi - vpadding;
                x_y = hstride * yy + hdilation * kwi - hpadding;
                if ((0 <= x_x) && (x_x < h) && (0 <= x_y) && (x_y < w)){{
                    val_k = *(SHIFT_POINTER(k, c, h, w, 0, cc, khi, kwi));
                    val_x = *(SHIFT_POINTER(x, c, h, w, nn, cc, x_x, x_y));
                    *(SHIFT_POINTER(dw, c, h, w, 0, cc, khi, kwi)) = ({T}) (val_x * val_dy);
                    *(SHIFT_POINTER(dx, c, h, w, nn, cc, x_x, x_y)) += ({T}) (val_k * val_dy);
                }}
            }}
        }}
    }}
}}
"""
        _t = DTYPE2CTYPE[self.model.dtype]  # variable Type

        code = code.format(macros=_macros,
                           func_name=_func_name,
                           T=_t
                           )
        module = SourceModule(code).get_function(_func_name)

        return module
    # ---

    def cuda_bias_sum_fwd_depthwise_conv(self, _func_name: str = "bias_sum_fwd_depthwise_conv") -> Function:
        _t = DTYPE2CTYPE[self.model.dtype]  # variable Type

        code = \
"""
__global__ void {func_name}({T}* x, {T}* bias,
                            int co, int N,
                            int num_workers)
{{
    int idx;

    for(idx = blockIdx.x * blockDim.x + threadIdx.x; idx < N; idx += num_workers)
    {{
        *(x + idx) += *(bias + ( idx / (N/co) ) );
    }}
}}
"""

        code = code.format(func_name=_func_name,
                           T=_t
                           )
        module = SourceModule(code).get_function(_func_name)

        return module
    # ----
#########################################################################################################
