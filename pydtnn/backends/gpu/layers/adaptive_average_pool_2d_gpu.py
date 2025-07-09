#
#  This file is part of Python Distributed Training of Neural Networks (PyDTNN)
#
#  Copyright (C) 2025 Universitat Jaume I
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

from pydtnn.layers import AdaptiveAveragePool2D
from .layer_gpu import LayerGPU

# Import from AveragePool2DGPU
from ..libs import libcudnn as cudnn

# Import from AbstractPool2DLayerGPU
from pydtnn.tracers import PYDTNN_OPS_EVENT, PYDTNN_OPS_EVENTS, PYDTNN_EVENT_FINISHED, PYDTNN_OPS_EVENT_enum

from ..tensor_gpu import TensorGPU
from pydtnn.performance_models import im2col_time, col2im_time
from pydtnn.utils import decode_tensor, encode_tensor
# noinspection PyUnresolvedReferences
import pycuda.gpuarray as gpuarray
# noinspection PyUnresolvedReferences
from pycuda.compiler import SourceModule
# noinspection PyUnresolvedReferences
from pycuda.driver import Function

import numpy as np
from pydtnn.utils import PYDTNN_TENSOR_FORMAT
from pydtnn.model import Model

DICT_SUPPORTED_TYPES = {np.float32: "float", np.float64: "double"}

# NOTE: IT IS NECESSARY TO TEST THIS!!
# TODO: Test this layer.
class AdaptiveAveragePool2DGPU(LayerGPU, AdaptiveAveragePool2D):
    
    def initialize(self, prev_shape, x: TensorGPU) -> None:
        LayerGPU.initialize(self, prev_shape, x)
        AdaptiveAveragePool2D.initialize(self, prev_shape)        

        self.threads = min(self.model.batch_size, 1024)
        self.blocks = max(self.model.batch_size, 1024) // self.threads + 1
        self.cuda_forward_func = self.cuda_adaptive_average_pooling_fwd(dtype=self.model.dtype)
        self.cuda_backward_func = self.cuda_adaptive_average_pooling_bwd(dtype=self.model.dtype)

        self.initialize_pool_2d_gpu(prev_shape, x)        
    # --- END initialize --- #
    
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.y = None
    # --- END __init__ --- #

    def cuda_adaptive_average_pooling_fwd(self, dtype: np.dtype) -> Function:
        
        _FUNC_NAME = "cuda_adaptive_average_pooling_fwd"
        _T = DICT_SUPPORTED_TYPES[dtype] # variable Type        
        _MACRO_INDEX_FIRST_ELEMENT = "INDEX_FIRST_ELEMENT"
        _MACRO_INDEX_LAST_ELEMENT = "INDEX_LAST_ELEMENT"
        _MACRO_SHIFT_POINTER = "SHIFT_POINTER"
        _FULL_MACRO_SHIFT_POINTER = f"#define {_MACRO_SHIFT_POINTER}(n_idx, c_idx, c, h, w) ((n_idx * c + c_idx) * h * w)"
        _FULL_MACRO_INDEX_FIRST_ELEMENT = f"#define {_MACRO_INDEX_FIRST_ELEMENT}(index, dim_in, dim_out) (int) ((index * dim_in) / dim_out)"
        _FULL_MACRO_INDEX_LAST_ELEMENT = f"#define {_MACRO_INDEX_LAST_ELEMENT}(index, dim_in, dim_out) (int) ((((index + 1) * dim_in) + dim_out - 1) / dim_out)"

        self.model:Model # NOTE: This is only for the hints.
        if self.model.tensor_format == PYDTNN_TENSOR_FORMAT.NCHW:
            code = \
            """
            {full_macro_index_first_element}
            {full_macro_index_last_element}
            {full_macro_shift_pointer}
            
            __global__ void {func_name}(const {T}* x_p, {T}* pooled_x_p,
                                        int n, int c, int h, int w, 
                                        int new_h, int new_w) 
            {{
                int n_idx, c_idx;
                int wi, hi, i, j;
                int h_start, h_end, w_start, w_end, elements_h, elements;
                {T} add;

                n_idx = blockIdx.x * blockDim.x + threadIdx.x;
                c_idx = blockIdx.y * blockDim.y + threadIdx.y;

                if (n_idx > n || c_idx > c) return;
                
                x_p += {macro_shift_pointer}(n_idx, c_idx, c, h, w);
                pooled_x_p += {macro_shift_pointer}(n_idx, c_idx, c, new_h, new_w);

                for(hi = 0; hi < new_h; hi++)
                {{
                    h_start = {macro_index_first_element}(wi, h, new_h);
                    h_end = {macro_index_last_element}(wi, h, new_h);
                    elements_h = h_end - h_start;
                    
                    for(wi = 0; wi < new_w; wi++, pooled_x_p++)
                    {{
                        w_start = {macro_index_first_element}(wi, w, new_w);
                        w_end = {macro_index_last_element}(wi, w, new_w);
                        elements = elements_h * (w_end - w_start);

                        for(i = h_start, add = ({T}) 0.0; i < h_end; i++)
                            for(j = w_start; j < w_end; j++, x_p++)
                                add += ({T}) (*x_p);

                        (*pooled_x_p) = ({T}) (add / elements);
                    }}                    
                }}
            }}
            """
            # -- END cuda_adaptive_average_pooling_fwd_nchw --
        elif self.model.tensor_format == PYDTNN_TENSOR_FORMAT.NHWC:
            code = \
            """
            {full_macro_index_first_element}
            {full_macro_index_last_element}
            {full_macro_shift_pointer}
            
            __global__ void {func_name}(const {T}* x_p, {T}* pooled_x_p,
                                        int n, int c, int h, int w, 
                                        int new_h, int new_w) 
            {{
                int n_idx, c_idx;
                int wi, hi, i, j;
                int h_start, h_end, w_start, w_end, elements_h, elements;
                {T} add;

                n_idx = blockIdx.x * blockDim.x + threadIdx.x;
                c_idx = blockIdx.y * blockDim.y + threadIdx.y;

                if (n_idx > n || c_idx > c) return;
                
                x_p += {macro_shift_pointer}(n_idx, c_idx, c, h, w);
                pooled_x_p += {macro_shift_pointer}(n_idx, c_idx, c, new_h, new_w);

                for(hi = 0; hi < new_h; hi++)
                {{
                    h_start = {macro_index_first_element}(wi, h, new_h);
                    h_end = {macro_index_last_element}(wi, h, new_h);
                    elements_h = h_end - h_start;
                    
                    for(wi = 0; wi < new_w; wi++, pooled_x_p++)
                    {{
                        w_start = {macro_index_first_element}(wi, w, new_w);
                        w_end = {macro_index_last_element}(wi, w, new_w);
                        elements = elements_h * (w_end - w_start);

                        for(i = h_start, add = ({T}) 0.0; i < h_end; i++)
                            for(j = w_start; j < w_end; j++, x_p++)
                                add += ({T}) (*x_p);

                        (*pooled_x_p) = ({T}) (add / elements);
                    }}                    
                }}
            }}
            """
            # -- END cuda_adaptive_average_pooling_fwd_nhwc --
        else:
            NotImplementedError(f"{self.model.tensor_format} is not an implemented format.")

        code = code.format(full_macro_index_first_element = _FULL_MACRO_INDEX_FIRST_ELEMENT,
                           full_macro_index_last_element = _FULL_MACRO_INDEX_LAST_ELEMENT,
                           full_macro_shift_pointer = _FULL_MACRO_SHIFT_POINTER,
                           macro_index_first_element = _MACRO_INDEX_FIRST_ELEMENT,
                           macro_index_last_element = _MACRO_INDEX_LAST_ELEMENT,
                           macro_shift_pointer = _MACRO_SHIFT_POINTER,
                           func_name = _FUNC_NAME,
                           T = _T, 
                           )
        module = SourceModule(code).get_function(_FUNC_NAME)
        
        return module
    # --- END cuda_adaptive_average_pooling_fwd --- #


    def cuda_adaptive_average_pooling_bwd(self, dtype: np.dtype) -> Function:
        
        _FUNC_NAME = "cuda_adaptive_average_pooling_bwd"
        _T = DICT_SUPPORTED_TYPES[dtype] # variable Type        
        _MACRO_INDEX_FIRST_ELEMENT = "INDEX_FIRST_ELEMENT"
        _MACRO_INDEX_LAST_ELEMENT = "INDEX_LAST_ELEMENT"
        _MACRO_SHIFT_POINTER = "SHIFT_POINTER"
        _FULL_MACRO_SHIFT_POINTER = f"#define {_MACRO_SHIFT_POINTER}(n_idx, c_idx, c, h, w) ((n_idx * c + c_idx) * h * w)"
        _FULL_MACRO_INDEX_FIRST_ELEMENT = f"#define {_MACRO_INDEX_FIRST_ELEMENT}(index, dim_in, dim_out) (int) ((index * dim_in) / dim_out)"
        _FULL_MACRO_INDEX_LAST_ELEMENT = f"#define {_MACRO_INDEX_LAST_ELEMENT}(index, dim_in, dim_out) (int) ((((index + 1) * dim_in) + dim_out - 1) / dim_out)"

        self.model:Model # NOTE: This is only for the hints.
        if self.model.tensor_format == PYDTNN_TENSOR_FORMAT.NCHW:
            code = \
            """
            {full_macro_index_first_element}
            {full_macro_index_last_element}
            {full_macro_shift_pointer}
            
            __global__ void {func_name}(const {T}* dy, {T}* dx,
                                        int n, int c, int h, int w, 
                                        int new_h, int new_w) 
            {{
                int n_idx, c_idx;
                int wi, hi, i, j;
                int h_start, h_end, w_start, w_end, elements_h, elements;
                {T} add;

                n_idx = blockIdx.x * blockDim.x + threadIdx.x;
                c_idx = blockIdx.y * blockDim.y + threadIdx.y;

                if (n_idx > n || c_idx > c) return;
                
                dy += {macro_shift_pointer}(n_idx, c_idx, c, h, w);
                dx += {macro_shift_pointer}(n_idx, c_idx, c, new_h, new_w);

                for(hi = 0; hi < h; hi++)
                {{
                    h_start = {macro_index_first_element}(wi, new_h, h);
                    h_end = {macro_index_last_element}(wi, new_h, h);
                    elements_h = h_end - h_start;
                    
                    for(wi = 0; wi < w; wi++, dx++)
                    {{
                        w_start = {macro_index_first_element}(wi, new_w, w);
                        w_end = {macro_index_last_element}(wi, new_w, w);
                        elements = elements_h * (w_end - w_start);

                        for(i = h_start, add = ({T}) 0.0; i < h_end; i++)
                            for(j = w_start; j < w_end; j++, dy++)
                                add += ({T}) (*dy);

                        (*dx) = ({T}) (add / elements);
                    }}                    
                }}
            }}
            """
            # -- END cuda_adaptive_average_pooling_bwd_nchw --
        elif self.model.tensor_format == PYDTNN_TENSOR_FORMAT.NHWC:
            code = \
            """
            {full_macro_index_first_element}
            {full_macro_index_last_element}
            {full_macro_shift_pointer}
            
            __global__ void {func_name}(const {T}* dy, {T}* dx,
                                        int n, int c, int h, int w, 
                                        int new_h, int new_w) 
            {{
                int n_idx, c_idx;
                int wi, hi, i, j;
                int h_start, h_end, w_start, w_end, elements_h, elements;
                {T} add;

                n_idx = blockIdx.x * blockDim.x + threadIdx.x;
                c_idx = blockIdx.y * blockDim.y + threadIdx.y;

                if (n_idx > n || c_idx > c) return;
                
                dy += {macro_shift_pointer}(n_idx, c_idx, c, h, w);
                dx += {macro_shift_pointer}(n_idx, c_idx, c, new_h, new_w);

                for(hi = 0; hi < h; hi++)
                {{
                    h_start = {macro_index_first_element}(wi, new_h, h);
                    h_end = {macro_index_last_element}(wi, new_h, h);
                    elements_h = h_end - h_start;
                    
                    for(wi = 0; wi < w; wi++, dx++)
                    {{
                        w_start = {macro_index_first_element}(wi, new_w, w);
                        w_end = {macro_index_last_element}(wi, new_w, w);
                        elements = elements_h * (w_end - w_start);

                        for(i = h_start, add = ({T}) 0.0; i < h_end; i++)
                            for(j = w_start; j < w_end; j++, dy++)
                                add += ({T}) (*dy);

                        (*dx) = ({T}) (add / elements);
                    }}                    
                }}
            }}
            """
            # -- END cuda_adaptive_average_pooling_bwd_nhwc --
        else:
            NotImplementedError(f"{self.model.tensor_format} is not an implemented format.")

        code = code.format(full_macro_index_first_element = _FULL_MACRO_INDEX_FIRST_ELEMENT,
                           full_macro_index_last_element = _FULL_MACRO_INDEX_LAST_ELEMENT,
                           full_macro_shift_pointer = _FULL_MACRO_SHIFT_POINTER,
                           macro_index_first_element = _MACRO_INDEX_FIRST_ELEMENT,
                           macro_index_last_element = _MACRO_INDEX_LAST_ELEMENT,
                           macro_shift_pointer = _MACRO_SHIFT_POINTER,
                           func_name = _FUNC_NAME,
                           T = _T, 
                           )
        module = SourceModule(code).get_function(_FUNC_NAME)
        
        return module
    # --- END cuda_adaptive_average_pooling_bwd --- #

    def initialize_pool_2d_gpu(self, prev_shape, x):
        self.hi, self.wi, self.ci = decode_tensor(prev_shape, self.model.tensor_format)
        self.shape = encode_tensor((self.ho, self.wo, self.co), self.model.tensor_format)
        
        pooling_shape = (self.ho, self.wo, self.co)
        # Activations y
        y = gpuarray.empty((self.model.batch_size, *pooling_shape), self.model.dtype)
        self.y = TensorGPU(y, self.model.tensor_format, self.model.cudnn_dtype)

        # Derivative dx
        dx_gpu = gpuarray.empty(self.x.ary.shape, self.model.dtype)
        self.dx = TensorGPU(dx_gpu, self.model.tensor_format, self.model.cudnn_dtype)

        self.fwd_time = \
            im2col_time(m=self.co, n=(self.model.batch_size * self.ho * self.wo * self.ci),
                        cpu_speed=self.model.cpu_speed, memory_bw=self.model.memory_bw,
                        dtype=self.model.dtype)
        self.bwd_time = \
            col2im_time(m=self.co, n=(self.model.batch_size * self.ho * self.wo * self.ci),
                        cpu_speed=self.model.cpu_speed, memory_bw=self.model.memory_bw,
                        dtype=self.model.dtype)
        # --- END initialize_pool_2d_gpu --- #

    def forward(self, x: TensorGPU) -> TensorGPU:
        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, self.id * PYDTNN_OPS_EVENTS + PYDTNN_OPS_EVENT_enum.FORWARD_CUDNN)

        if self.pooling_not_needed:
            self.y = x
        else:
            if self.model.tensor_format == PYDTNN_TENSOR_FORMAT.NCHW:
                n, c, h, w = x.shape
            else:
                n, h, w, c = x.shape

            self.cuda_forward_func(x, self.y, n, c, h, w, self.ho, self.co,
                           grid=(self.blocks, 1, 1), block=(self.threads, 1, 1),
                           stream=self.model.stream)

        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, PYDTNN_EVENT_FINISHED)
        return self.y
    # --- END forward --- #

    def backward(self, dy: TensorGPU) -> TensorGPU:
        
        if self.model.tensor_format == PYDTNN_TENSOR_FORMAT.NCHW:
            n, c, _, _ = dy.shape
        else:
            n, _, _, c = dy.shape

        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, self.id * PYDTNN_OPS_EVENTS + PYDTNN_OPS_EVENT_enum.BACKWARD_CUDNN_DX)
        # Compute dx
        self.cuda_forward_func(dy, self.dx, n, c, self.ho, self.co, self.hi, self.wi,
                        grid=(self.blocks, 1, 1), block=(self.threads, 1, 1),
                        stream=self.model.stream)
        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, PYDTNN_EVENT_FINISHED)
        return self.dx
    # --- END backward --- #
    # END of methods from AbstractPool2DLayerGPU #

# --- END AdaptiveAveragePool2DGPU --- #
