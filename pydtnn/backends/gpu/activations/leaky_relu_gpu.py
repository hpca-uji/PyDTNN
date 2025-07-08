#
#  This file is part of Python Distributed Training of Neural Networks (PyDTNN)
#
#  Copyright (C) 2021-25 Universitat Jaume I
#
#  PyDTNN is free software: you can redistribute it and/or modify it under the
#  terms of the GNU General Public License as published by the Free Software
#  Foundation, either version 3 of the License, or (at your option) any later
#  version.
#
#  This program is distributed in the hope that it will be useful, but WITHOUT
#  ANY WARRANTY; without even the implied warranty of MERCHANTABILITY
#  or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public
#  License for more details.
#
#  You should have received a copy of the GNU General Public License along
#  with this program. If not, see <https://www.gnu.org/licenses/>.
#

from pydtnn.activations.leaky_relu import LeakyRelu
from pydtnn.performance_models import im2col_time, col2im_time
from pydtnn.utils import decode_tensor
from ..tensor_gpu import TensorGPU
from pydtnn.backends.gpu.activations.activation_gpu import ActivationGPU

import numpy as np
from pydtnn.tracers import PYDTNN_OPS_EVENT, PYDTNN_OPS_EVENTS, PYDTNN_OPS_FORWARD_CUDNN, PYDTNN_OPS_BACKWARD_CUDNN_DX
# noinspection PyUnresolvedReferences
import pycuda.gpuarray as gpuarray
# noinspection PyUnresolvedReferences
from pycuda.compiler import SourceModule
# noinspection PyUnresolvedReferences
from pycuda.driver import Function


DICT_SUPPORTED_TYPES = {np.float32: "float", np.float64: "double"}

class LeakyReluGPU(LeakyRelu, ActivationGPU):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.mask: TensorGPU | None = None
        self.y: TensorGPU | None = None
    # --- END __init__ --- #
    
    def initialize(self, prev_shape: tuple[int, ...], x: TensorGPU) -> None:
        ActivationGPU.initialize(self, prev_shape, x)
        LeakyRelu.initialize(self, prev_shape)        

        self.threads = min(self.model.batch_size, 1024)
        self.blocks = max(self.model.batch_size, 1024) // self.threads + 1
        self.cuda_fwd_func = self.cuda_adaptive_average_pooling_fwd(dtype=self.model.dtype)
        self.cuda_bwd_func = self.cuda_adaptive_average_pooling_bwd(dtype=self.model.dtype)
        
        self.grid = (self.blocks, 1, 1)
        self.block = (self.threads, 1, 1)

        self.total_num_threads = np.prod(self.grid, dtype=np.int32) * np.prod(self.block, dtype=np.int32)

        self.initialize_relu_2d_gpu(prev_shape)
    # --- END initialize --- #

    def cuda_adaptive_average_pooling_fwd(self, dtype: np.dtype) -> Function:
        _func_name = "cuda_leaky_relu_fwd"
        _t = DICT_SUPPORTED_TYPES[dtype] # variable Type

        code = \
"""
__global__ void {func_name}({T}* x, {T}* max, {T}* mask, 
                            float negative_slope, int num_workers, int N)
{{
    int idx, i;
    {T} elem;

    for(i = blockIdx.x * blockDim.x + threadIdx.x; i < N; i += num_workers)
    {{
        elem = x[i];

        if (elem > 0)
        {{
            max[i] = elem;
            mask[i] = 1;
        }}
        else if(elem < 0)
        {{
            max[i] = ({T}) (elem * negative_slope);
            mask[i] = negative_slope;
        }}
        else
        {{
            max[i] = 0;
            mask[i] = 0;
        }}
    }}
}}
"""
        code = code.format(func_name = _func_name, T = _t)

        return SourceModule(code).get_function(_func_name)
    # --- END cuda_adaptive_average_pooling_fwd --- #

    def cuda_adaptive_average_pooling_bwd(self, dtype: np.dtype) -> Function:
        _func_name = "cuda_leaky_relu_bwd"
        _t = DICT_SUPPORTED_TYPES[dtype] # variable Type

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
        code = code.format(func_name = _func_name, T = _t)

        return SourceModule(code).get_function(_func_name)
    # --- END cuda_adaptive_average_pooling_bwd --- #


    def forward(self, x: TensorGPU) -> TensorGPU:
        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, self.id * PYDTNN_OPS_EVENTS + PYDTNN_OPS_FORWARD_CUDNN)
        
        n = np.prod(x.shape, dtype=np.int32)

        self.cuda_fwd_func(x.ary, self.mask.ary, self.max.ary,
                           np.float32(self.negative_slope), self.total_num_threads, n,
                           grid=self.grid, block=self.block, stream=self.model.stream)

        self.y: TensorGPU = self.mask

        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, 0)

        return self.y
    # --- END forward --- #

    def backward(self, dy: TensorGPU) -> TensorGPU:
        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, self.id * PYDTNN_OPS_EVENTS + PYDTNN_OPS_BACKWARD_CUDNN_DX)

        n = np.prod(dy.shape, dtype=np.int32)

        self.cuda_bwd_func(self.dx.ary, dy.ary, self.mask.ary,
                            self.total_num_threads, n,
                            grid=self.grid, block=self.block,
                            stream=self.model.stream)

        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, 0)
        return self.dx
    # --- END backward --- #

    def initialize_relu_2d_gpu(self, prev_shape: tuple[int, ...]) -> None:
        self.hi, self.wi, self.ci = decode_tensor(prev_shape, self.model.tensor_format)
        self.shape = prev_shape
        
        n:int = self.model.batch_size * self.hi * self.wi * self.ci
        
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
    # --- END initialize_pool_2d_gpu --- #
