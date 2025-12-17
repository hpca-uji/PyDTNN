import numpy as np

from pydtnn.tracers.events import PYDTNN_OPS_EVENT, PYDTNN_OPS_EVENTS, PYDTNN_EVENT_FINISHED, PYDTNN_OPS_EVENT_enum
from pydtnn.backends.gpu.layers.conv_2d import Conv2DGPU, MACROS_NCHW, MACROS_NHWC
from pydtnn.backends.gpu.utils.tensor_gpu import TensorGPU
from pydtnn.utils.constants import ArrayShape, DTYPE2CTYPE

from pydtnn.utils.tensor import TensorFormat, format_transpose
from typing import Any, override

import pycuda.gpuarray as gpuarray  #type: ignore
from pycuda.compiler import SourceModule  #type: ignore
from pycuda.driver import Function  #type: ignore

class Conv2DPointwiseGPU(Conv2DGPU):

    def _initializing_special_parameters(self):
        self.kh = self.kw = 1
        # Setting weights
        match self.model.tensor_format:
                case TensorFormat.NCHW:
                    self.weights_shape = (self.co, self.ci)
                case TensorFormat.NHWC:
                    self.weights_shape = (self.co, self.ci)
                case _:
                    raise NotImplementedError(f"\"{self.model.tensor_format}\" format not implemented.")
        #--

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
            case TensorFormat.NHWC:
                func_name = "cuda_depthwise_conv_2d_{fwd_bwd}_nhwc"
                macros = MACROS_NHWC
                self.bias_sum_bwd = self.cuda_sum_bias_axis_012()
            case _:
                raise NotImplementedError(f"\"conv_2d_gpu_depthwise\" is not implemented for \"{self.model.tensor_format}\" format.")

        self.total_num_threads = np.prod(self.grid, dtype=np.int32) * np.prod(self.block, dtype=np.int32)

        y_gpu = gpuarray.to_gpu(np.zeros(shape=(self.model.batch_size, *self.shape), dtype=self.model.dtype))
        self.y = TensorGPU(y_gpu, self.model.tensor_format, self.model.cudnn_dtype)

        dx_gpu = gpuarray.zeros((self.model.batch_size, *self.shape), self.model.dtype)
        self.dx = TensorGPU(dx_gpu, self.model.tensor_format, self.model.cudnn_dtype)

        self.forward = self._forward_pointwise
        self.backward = self._backward_pointwise
        self.fwd_func: Function = self.cuda_pointwise_conv_2d_fwd(func_name.format(fwd_bwd="fwd"), macros)
        self.bwd_func: Function = self.cuda_pointwise_conv_2d_bwd(func_name.format(fwd_bwd="bwd"), macros)
        self.bias_sum_fwd: Function = self.cuda_bias_pointwise_conv_2d_fwd("bias_pointwise_conv_2d_fwd", macros)
    # ----

    def _forward_pointwise(self, x: TensorGPU) -> TensorGPU:

        self.x = x
        self.y.fill(0)

        n, c, h, w = self.model.decode_shape(x.shape)  # type: ignore (it's okay)

        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, self.id * PYDTNN_OPS_EVENTS + PYDTNN_OPS_EVENT_enum.FORWARD_CUDNN)
        self.fwd_func(x.ary, self.weights.ary, self.y.ary,
                      np.int32(n), np.int32(c), np.int32(h), np.int32(w),
                      np.int32(self.co), self.total_num_threads,
                      grid=self.grid, block=self.block, stream=self.model.stream)

        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, PYDTNN_EVENT_FINISHED)

        if self.use_bias:
            self.biases: TensorGPU
            self.model.tracer.emit_event(PYDTNN_OPS_EVENT, self.id * PYDTNN_OPS_EVENTS + PYDTNN_OPS_EVENT_enum.FORWARD_CUDNN_SUM_BIASES)
            self.bias_sum_fwd(x.ary, self.biases.ary,
                              np.int32(n), np.int32(c), np.int32(h), np.int32(w),
                              np.int32(n * c * h * w),
                              self.total_num_threads,
                              grid=self.grid, block=self.block, stream=self.model.stream)
            self.model.tracer.emit_event(PYDTNN_OPS_EVENT, PYDTNN_EVENT_FINISHED)

        return self.y
    # -----

    def _backward_pointwise(self, dy: TensorGPU) -> TensorGPU:
        n, c, h, w = self.model.decode_shape(dy.shape)  # type: ignore (it's okay)
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

    @override
    def _export_weights_dw(self, key: str) -> Any:
        value = getattr(self, key)

        match self.model.tensor_format:
            case TensorFormat.NHWC:
                # NHWC's src: self.ci, self.co
                # NCHW's dst: self.co, self.ci
                gpu_ary = value.ary
                cpu_ary = gpu_ary.get()
                return np.asarray(format_transpose(np.squeeze(cpu_ary, axis=(1, 2)), "IO", "OI"), dtype=np.float64, order="C", copy=True)
            case TensorFormat.NCHW:
                # NHWC's src: self.ci, self.co
                # NCHW's dst: self.co, self.ci
                gpu_ary = value.ary
                cpu_ary = gpu_ary.get()
                return np.asarray(np.squeeze(cpu_ary, axis=(2, 3)), dtype=np.float64, order="C", copy=True)
            case default:
                return super()._export_prop(key)
    # ------

    @override
    def _import_weights_dw(self, key: str, value: Any) -> None:
        attribute = getattr(self, key)
        match self.model.tensor_format:
            case TensorFormat.NHWC:
                # NHWC's src: self.ci, self.co
                # NCHW's dst: self.co, self.ci
                cpu_ary = np.asarray(np.expand_dims(format_transpose(value, "OI", "IO"), axis=(1, 2)), dtype=self.model.dtype, order="C", copy=None)
                attribute.ary.set(cpu_ary)
                return
            case TensorFormat.NCHW:
                gpu_ary = attribute.ary
                cpu_ary = np.asarray(np.expand_dims(value, axis=(2, 3)), dtype=self.model.dtype, order="C", copy=None)
                gpu_ary.set(cpu_ary)
                return 
            case default:
                return super()._export_prop(key)
    # ---

#########################################################################################################
## CUDA CODE ##
###############
    def cuda_pointwise_conv_2d_fwd(self, _func_name: str, _macros: str) -> Function:

        code = \
"""
{macros}

__global__ void {func_name}({T}* x, {T}* k, {T}* y,
                            int n, int c, int h, int w,
                            int yc, int num_workers)
{{
    int idx, ni, ci, hi, wi, yci;
    {T} val_k, val_x;

    int N = n*c*h*w;

    // k.shape = (yc, x's c)

    for(idx = blockIdx.x * blockDim.x + threadIdx.x; idx < N; idx += num_workers)
    {{
        ni = INDEX_N(idx, N, n);
        ci = INDEX_C(idx, c, h, w);
        hi = INDEX_H(idx, c, h, w);
        wi = INDEX_W(idx, c, h, w);

        val_x = *(SHIFT_POINTER(x, c, h, w, ni, ci, hi, wi));
        for(yci = 0; yci < yc; yci++)
        {{
            //y = x * k
            //val_k = k[yci][ci]; ==> val_k = k + (yci * c + ci);
            //val_k = k[ci][yci]; ==> val_k = k + (ci * kc + yci);
            val_k = *(SHIFT_POINTER_K(k, c, yc, ci, yci));
            *(SHIFT_POINTER(y, yc, h, w, ni, yci, hi, wi)) += ({T}) (val_x * val_k);
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

    def cuda_pointwise_conv_2d_bwd(self, _func_name: str, _macros: str) -> Function:

        code = \
"""
{macros}

__global__ void {func_name}({T}* dy, {T}* x, {T}* k,
                            {T}* dx, {T}* dw,
                            int n, int c, int h, int w,
                            int xc, int num_workers)
{{
    int idx, ni, ci, hi, wi, xci;
    {T} val_dy, val_k, val_x;

    int N = n*c*h*w;

    // NCHW: k.shape = dw.shape = (dy's c , x's c)
    // NHWC: k.shape = dw.shape = (x's c, dy's c)

    for(idx = blockIdx.x * blockDim.x + threadIdx.x; idx < N; idx += num_workers)
    {{
        ni = INDEX_N(idx, N, n);
        ci = INDEX_C(idx, c, h, w);
        hi = INDEX_H(idx, c, h, w);
        wi = INDEX_W(idx, c, h, w);

        val_dy = *SHIFT_POINTER(dy, c, h, w, ni, ci, hi, wi);
        for(xci = 0; xci < kc; xci++)
        {{
            //dw = x * dy
            val_x = *(SHIFT_POINTER(x, xc, h, w, ni, xci, hi, wi));
            *(SHIFT_POINTER_K(dw, c, xc, ci, xci)) = ({T}) (val_x * val_dy);

            //dx = w * dy
            val_k = *(SHIFT_POINTER_K(k, c, xc, ci, xci));
            *(SHIFT_POINTER(dx, kc, h, w, nn, xci, hi, wi)) += ({T}) (val_k * val_dy);
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

    def cuda_bias_pointwise_conv_2d_fwd(self, _func_name: str, _macros: str) -> Function:

        code = \
"""
{macros}

__global__ void {func_name}({T}* y, {T}* b,
                            int n, int c, int h, int w,
                            int N,
                            int num_workers)
{{
    int idx, ni, ci, hi, wi;

    // self.biases.shape = (self.co,)

    for(idx = blockIdx.x * blockDim.x + threadIdx.x; idx < N; idx += num_workers)
    {{
        ni = INDEX_N(idx, N, n);
        ci = INDEX_C(idx, c, h, w);
        hi = INDEX_H(idx, c, h, w);
        wi = INDEX_W(idx, c, h, w);

        *(SHIFT_POINTER(y, c, h, w, ni, ci, hi, wi)) += (*(b+ci));
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
#########################################################################################################
