from pydtnn.activations.relu6 import Relu6
from pydtnn.utils.performance_models import im2col_time, col2im_time
from pydtnn.backends.gpu.utils.tensor_gpu import TensorGPU
from pydtnn.backends.gpu.activations.activation import ActivationGPU
from pydtnn.utils.constants import ArrayShape, DTYPE2CTYPE

import numpy as np
from pydtnn.tracers.events import PYDTNN_OPS_EVENT, PYDTNN_OPS_EVENTS, PYDTNN_OPS_EVENT_enum
import pycuda.gpuarray as gpuarray  # type: ignore
from pycuda.compiler import SourceModule  # type: ignore
from pycuda.driver import Function  # type: ignore


class Relu6GPU(Relu6[TensorGPU], ActivationGPU):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.mask: TensorGPU = None  # type: ignore
        self.y: TensorGPU = None  # type: ignore
    # ---

    def initialize(self, prev_shape: ArrayShape, x: TensorGPU) -> None:
        super().initialize(prev_shape, x)

        self.cuda_fwd_func = self.cuda_adaptive_average_pooling_fwd(dtype=self.model.dtype)
        self.cuda_bwd_func = self.cuda_adaptive_average_pooling_bwd(dtype=self.model.dtype)

        self.total_num_threads = np.prod(self.grid, dtype=np.int32) * np.prod(self.block, dtype=np.int32)

        self.initialize_relu_2d_gpu(prev_shape)
    # ----

    def cuda_adaptive_average_pooling_fwd(self, dtype: np.dtype) -> Function:
        _func_name = "cuda_relu6_fwd"
        _t = DTYPE2CTYPE[dtype]  # variable Type

        code = \
            """
__global__ void {func_name}({T}* x, {T}* max, {T}* mask,
                            float cap, int num_workers, int N)
{{
    int i;
    {T} elem;

    for(i = blockIdx.x * blockDim.x + threadIdx.x; i < N; i += num_workers)
    {{
        elem = x[i];

        if(elem >= cap)
        {{
            max[i] = ({T}) cap;
            mask[i] = 1;
        }}
        else if (elem > 0)
        {{
            max[i] = elem;
            mask[i] = 1;
        }}
        else
        {{
            max[i] = 0;
            mask[i] = 0;
        }}
    }}
}}
"""
        code = code.format(func_name=_func_name, T=_t)

        return SourceModule(code).get_function(_func_name)
    # ----

    def cuda_adaptive_average_pooling_bwd(self, dtype: np.dtype) -> Function:
        _func_name = "cuda_relu6_bwd"
        _t = DTYPE2CTYPE[dtype]  # variable Type

        code = \
            """
__global__ void {func_name}({T}* dx, {T}* dy, {T}* mask,
                            int num_workers, int N)
{{
    int i;

    for(i = blockIdx.x * blockDim.x + threadIdx.x; i < N; i += num_workers)
        dx[i] = dy[i] * mask[i];
}}
"""
        code = code.format(func_name=_func_name, T=_t)

        return SourceModule(code).get_function(_func_name)
    # ----

    def forward(self, x: TensorGPU) -> TensorGPU:
        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, self.id * PYDTNN_OPS_EVENTS + PYDTNN_OPS_EVENT_enum.FORWARD_CUDNN)

        n = np.prod(x.shape, dtype=np.int32)

        self.cuda_fwd_func(x.ary, self.mask.ary, self.max.ary,
                           np.float32(self.cap), self.total_num_threads, n,
                           grid=self.grid, block=self.block, stream=self.model.stream)

        self.y: TensorGPU = self.mask

        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, 0)

        return self.y

    def backward(self, dy: TensorGPU) -> TensorGPU:
        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, self.id * PYDTNN_OPS_EVENTS + PYDTNN_OPS_EVENT_enum.BACKWARD_CUDNN_DX)

        n = np.prod(dy.shape, dtype=np.int32)

        self.cuda_bwd_func(self.dx.ary, dy.ary, self.mask.ary,
                           self.total_num_threads, n,
                           grid=self.grid, block=self.block,
                           stream=self.model.stream)

        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, 0)
        return self.dx

    def initialize_relu_2d_gpu(self, prev_shape: ArrayShape) -> None:
        self.ci, self.hi, self.wi = self.model.decode_shape(prev_shape)
        self.shape = prev_shape

        n: int = self.model.batch_size * self.hi * self.wi * self.ci

        _max = gpuarray.empty((self.model.batch_size, *self.shape), self.model.dtype)
        self.max = TensorGPU(_max, self.model.tensor_format, self.model.cudnn_dtype)

        _mask = gpuarray.empty((self.model.batch_size, *self.shape), self.model.dtype)
        self.mask = TensorGPU(_mask, self.model.tensor_format, self.model.cudnn_dtype)

        # Derivative dx
        dx_gpu = gpuarray.empty((self.model.batch_size, *self.shape), self.model.dtype)
        self.dx = TensorGPU(dx_gpu, self.model.tensor_format, self.model.cudnn_dtype)

        self.fwd_time = \
            im2col_time(m=self.ci, n=n, cpu_speed=self.model.cpu_speed,
                        memory_bw=self.model.memory_bw, dtype=self.model.dtype)
        self.bwd_time = \
            col2im_time(m=self.ci, n=n, cpu_speed=self.model.cpu_speed,
                        memory_bw=self.model.memory_bw, dtype=self.model.dtype)
