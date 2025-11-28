from typing import Any
from pydtnn.layers.conv_2d import Conv2D

import pycuda.driver as drv  #type: ignore
import pycuda.gpuarray as gpuarray  #type: ignore
from pycuda.compiler import SourceModule  #type: ignore
from pycuda.driver import Function  #type: ignore

import numpy as np

from pydtnn.utils.performance_models import matmul_time
from pydtnn.backends.gpu.layers.layer import LayerGPU

from pydtnn.backends.gpu.utils.tensor_gpu import TensorGPU
from pydtnn.utils.tensor import TensorFormat
from pydtnn.utils.constants import ArrayShape, DTYPE2CTYPE, Parameters

class Conv2DGPU(Conv2D[TensorGPU], LayerGPU):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # The following attributes will be initalized later.
        self.fwd_algo: int = None  #type: ignore
        self.bwd_dw_algo: int = None  #type: ignore
        self.bwd_dx_algo: int = None  #type: ignore
        self.conv_desc = None
    # ----

    def initialize(self, prev_shape: ArrayShape, x: TensorGPU) -> None:
        super().initialize(prev_shape, x)

        self.stream_2 = drv.Stream()

        self.weights_cpu = self.weights_initializer(self.weights_shape, self.model.dtype)
        weights_gpu = gpuarray.to_gpu(self.weights_cpu)
        self.weights = TensorGPU(weights_gpu, self.model.tensor_format, self.model.cudnn_dtype, TensorGPU.TensorTypeEnum.FILTER)
        # Biases
        if self.use_bias:
            biases_shape = self.model.encode_shape((1, self.co, 1, 1))
            self.biases_cpu = self.biases_initializer(biases_shape, self.model.dtype)
            biases_gpu = gpuarray.to_gpu(self.biases_cpu)
            self.biases = TensorGPU(biases_gpu, self.model.tensor_format, self.model.cudnn_dtype)

        self.fwd_time = \
            matmul_time(m=self.co, n=(self.model.batch_size * self.ho * self.wo), k=(self.ci * self.kh * self.kw),
                        cpu_speed=self.model.cpu_speed, memory_bw=self.model.memory_bw, dtype=self.model.dtype) 
        self.bwd_time = \
            matmul_time(m=self.co, n=(self.ci * self.kh * self.kw), k=(self.model.batch_size * self.ho * self.wo),
                        cpu_speed=self.model.cpu_speed, memory_bw=self.model.memory_bw, dtype=self.model.dtype) + \
            matmul_time(m=(self.ci * self.kh * self.kw), n=(self.model.batch_size * self.ho * self.wo), k=self.co,
                        cpu_speed=self.model.cpu_speed, memory_bw=self.model.memory_bw, dtype=self.model.dtype)  # type: ignore (It is correct.)

        if self.model.gpudirect:
            bias_tensor_type = TensorGPU.TensorTypeEnum.FILTER
            _drv = drv
        else:
            bias_tensor_type = TensorGPU.TensorTypeEnum.TENSOR
            _drv = None

        # Derivative dw and derivative db
        self.dw_cpu, self.dw = TensorGPU.initialize(self.weights.ary.shape, self.model.dtype, tensor_format=self.model.tensor_format,
                                                    cudnn_dtype=self.model.cudnn_dtype, gpudirect=self.model.gpudirect, 
                                                    tensor_type=TensorGPU.TensorTypeEnum.FILTER, drv=_drv)
        if self.use_bias:
            self.biases: TensorGPU
            self.db_cpu, self.db = TensorGPU.initialize(self.biases.ary.shape, self.model.dtype, tensor_format=self.model.tensor_format,
                                                        cudnn_dtype=self.model.cudnn_dtype, gpudirect=self.model.gpudirect, 
                                                        tensor_type=bias_tensor_type, drv=_drv)
    # ----

    def _export_weights_dw(self, key: str) -> Any:
        # NOTE: Every variant must implement their version of this method.
        #super()._export_prop(key)
        msg = "This is a \"fake\" function. It must be overrided by the child classes."
        raise NotImplementedError(f"Conv2DGPU forward: {msg}")
    # ----

    def _export_biases_db(self, key: str) -> Any:
        value = getattr(self, key)
        gpu_ary = value.ary
        cpu_ary = gpu_ary.get()

        match self.model.tensor_format:
            case TensorFormat.NHWC:
                return np.asarray(np.squeeze(cpu_ary, axis=(0, 1, 2)), dtype=np.float64, order="C", copy=True)
            case TensorFormat.NCHW:
                return np.asarray(np.squeeze(cpu_ary, axis=(0, 2, 3)), dtype=np.float64, order="C", copy=True)
            case _:
                return super()._export_prop(key)
    # ----

    def _export_prop(self, key: str) -> Any:
        match key:
            case Parameters.WEIGHTS | Parameters.DW:
                return self._export_weights_dw(key)
            case Parameters.BIASES | Parameters.DB:
                return self._export_biases_db(key)
            case _:
                return super()._export_prop(key)
    # ----

    def _import_biases_db(self, key: str, value: Any) -> None:
        attribute = getattr(self, key)
        
        match self.model.tensor_format:
            case TensorFormat.NHWC:
                cpu_ary = np.asarray(np.expand_dims(value, axis=(0, 1, 2)), dtype=self.model.dtype, order="C", copy=None)
                attribute.ary.set(cpu_ary)
                return
            case TensorFormat.NCHW:
                cpu_ary = np.asarray(np.expand_dims(value, axis=(0, 2, 3)), dtype=self.model.dtype, order="C", copy=None)
                attribute.ary.set(cpu_ary)
                return
            # case default: (next return)
        return super()._import_prop(key, value)
    # ----

    def _import_weights_dw(self, key: str, value: Any) -> None:
        # NOTE: Every variant must implement their version of this method.
        #super()._export_prop(key)
        msg = "This is a \"fake\" function. It must be overrided by the child classes"
        raise NotImplementedError(f"Conv2DGPU forward: {msg}")
    # ----

    def _import_prop(self, key: str, value) -> None:
        match key:
            case Parameters.WEIGHTS | Parameters.DW:
                return self._import_weights_dw(key, value)
            case Parameters.BIASES | Parameters.DB:
                return self._import_biases_db(key, value)
            # 
            case _:
                return super()._import_prop(key, value)
    # ----

    def forward(self, x: TensorGPU) -> TensorGPU:
        msg = "This is a fake forward function. It must be masked on initialization by a _forward implementation."
        raise NotImplementedError(f"Conv2DGPU forward: {msg}")

    def backward(self, dy: TensorGPU) -> TensorGPU:
        msg = "This is a fake backward function. It must be masked on initialization by a _backward implementation."
        raise NotImplementedError(f"Conv2DGPU backward: {msg}")


#########################################################################################################
## CUDA-RELATED COMMON CODE ##
##############################
    def cuda_sum_bias_axis_023(self, _func_name: str = "bias_sum_bwd_depthwise_conv_nchw") -> Function:
        _t = DTYPE2CTYPE[self.model.dtype]  # variable Type

        # np.sum(dy, axis=(0, 2, 3), out=self.db)
        code = \
            """
__global__ void {func_name}({T}* dy, {T}* db
                            int c, int h, int w,
                            int N, int num_workers)
{{
    int idx, index_c;

    for(idx = blockIdx.x * blockDim.x + threadIdx.x; idx < N; idx += num_workers)
    {{
        index_c = (idx / (h*w)) % c;
        *(db + index_c) += *(dy + idx);
    }}
}}
"""

        code = code.format(func_name=_func_name,
                           T=_t
                           )
        module = SourceModule(code).get_function(_func_name)

        return module
    # ----

    def cuda_sum_bias_axis_012(self, _func_name: str = "bias_sum_bwd_depthwise_conv_nhwc") -> Function:
        _t = DTYPE2CTYPE[self.model.dtype]  # variable Type

        # np.sum(dy, axis=(0, 1, 2), out=self.db)
        code = \
            """
__global__ void {func_name}({T}* dy, {T}* db,
                            int c, int N,
                            int num_workers)
{{
    int idx;

    for(idx = blockIdx.x * blockDim.x + threadIdx.x; idx < N; idx += num_workers)
    {{
        *(db + (idx % c)) += *(dy + idx);
    }}
}}
"""

        code = code.format(func_name=_func_name,
                           T=_t
                           )
        module = SourceModule(code).get_function(_func_name)

        return module
    # ----

# CUDA-related constans

MACROS_NCHW = \
    """
#define SHIFT_POINTER(p, c, h, w, ni, ci, hi, wi) p + ((ni * c + ci) * h + hi) * w + wi
#define SHIFT_POINTER_K(p, c, yc, ci, yci) p + (yci * c + ci)
#define INDEX_N(idx, N, n) idx * n / N
#define INDEX_C(idx, c, h, w) (idx / (h * w)) % c
#define INDEX_H(idx, c, h, w) (idx / w) % h
#define INDEX_W(idx, c, h, w) idx % w
"""
# ---

MACROS_NHWC = \
    """
#define SHIFT_POINTER(p, c, h, w, ni, ci, hi, wi) p + ((ni * h + hi) * w + wi) * c + ci
#define SHIFT_POINTER_K(p, c, yc, ci, yci) p + (ci * yc + yci)
#define INDEX_N(idx, N, n) idx * n / N
#define INDEX_H(idx, h, w, c) (idx / (w * c)) % h
#define INDEX_W(idx, h, w, c) (idx / c) % w
#define INDEX_C(idx, h, w, c) idx % c
"""
# ---
#########################################################################################################
