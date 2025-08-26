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

# noinspection PyUnresolvedReferences
from pydtnn.layers import Conv2D
from ..libs import libcudnn as cudnn
# noinspection PyUnresolvedReferences
import pycuda.driver as drv
# noinspection PyUnresolvedReferences
import pycuda.gpuarray as gpuarray
# noinspection PyUnresolvedReferences
from pycuda.compiler import SourceModule
# noinspection PyUnresolvedReferences
from pycuda.driver import Function

from pydtnn.performance_models import *
from pydtnn.tracers import PYDTNN_OPS_EVENT, PYDTNN_OPS_EVENTS, PYDTNN_EVENT_FINISHED, PYDTNN_OPS_EVENT_enum
from .layer_gpu import LayerGPU
from .memory_allocation import checkConvolutionMemory, getConvolutionWorkspaceSize, getConvolutionWorkspacePtr
from ..tensor_gpu import TensorGPU
from pydtnn.utils import PYDTNN_TENSOR_FORMAT
from pydtnn.layers.conv_2d import GroupingEnum

DICT_SUPPORTED_TYPES = {np.float32: "float", np.float64: "double"}

class Conv2DGPU(LayerGPU, Conv2D):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.fwd_algo = None
        self.fwd_time = None
        self.bwd_dw_algo = None
        self.bwd_dx_algo = None
        self.conv_desc = None

        # NOTE: Seems that in PyDTNN, usually the ".x" (blockIdx.x, threadIdx.x, ...) is the only dimension used.
        self.grid = (self.blocks, 1, 1)
        self.block = (self.threads, 1, 1)
    # ---

    def initialize(self, prev_shape: tuple[int, ...], x: TensorGPU) -> TensorGPU:
        super().initialize(prev_shape, x)
        # This weight shape is required for cuDNN when NHWC is seleted!
        if self.model.tensor_format is PYDTNN_TENSOR_FORMAT.NHWC:
            self.weights_shape = (self.co, *self.filter_shape, self.ci)

        self.stream_2 = drv.Stream()

        self.weights_cpu = self.weights_initializer(self.weights_shape, self.model.dtype)
        weights_gpu = gpuarray.to_gpu(self.weights_cpu)
        self.weights = TensorGPU(weights_gpu, self.model.tensor_format, self.model.cudnn_dtype, "filter")
        # Biases
        if self.use_bias:
            self.biases_cpu = self.biases_initializer((1, self.co, 1, 1) \
               if self.model.tensor_format is PYDTNN_TENSOR_FORMAT.NCHW else (1, 1, 1, self.co), self.model.dtype)
            biases_gpu = gpuarray.to_gpu(self.biases_cpu)
            self.biases = TensorGPU(biases_gpu, self.model.tensor_format, self.model.cudnn_dtype)

        self.fwd_time = \
            matmul_time(m=self.co, n=(self.model.batch_size * self.ho * self.wo), k=(self.ci * self.kh * self.kw),
                        cpu_speed=self.model.cpu_speed, memory_bw=self.model.memory_bw, dtype=self.model.dtype)
        self.bwd_time = \
            matmul_time(m=self.co, n=(self.ci * self.kh * self.kw), k=(self.model.batch_size * self.ho * self.wo),
                        cpu_speed=self.model.cpu_speed, memory_bw=self.model.memory_bw, dtype=self.model.dtype) + \
            matmul_time(m=(self.ci * self.kh * self.kw), n=(self.model.batch_size * self.ho * self.wo), k=self.co,
                        cpu_speed=self.model.cpu_speed, memory_bw=self.model.memory_bw, dtype=self.model.dtype)

        match self.grouping:
            case GroupingEnum.STANDARD:
                self.initialize_standard_grouping(x)
            case GroupingEnum.DEPTHWISE:
                self.initialize_depthwise_grouping()
            case GroupingEnum.POINTWISE:
                self.initialize_pointwise_grouping()
    # -----

    ####################
    ## STANDARD CONV. ##
    ####################

    def initialize_standard_grouping(self, x: TensorGPU):

        # This weight shape is required for cuDNN when NHWC is seleted!
        if self.model.tensor_format is PYDTNN_TENSOR_FORMAT.NHWC:
            self.weights_shape = (self.co, *self.filter_shape, self.ci)
        
        self.weights_cpu = self.weights_initializer(self.weights_shape, self.model.dtype)
        weights_gpu = gpuarray.to_gpu(self.weights_cpu)
        self.weights = TensorGPU(weights_gpu, self.model.tensor_format, self.model.cudnn_dtype, "filter")

        # Activations y
        y_gpu = gpuarray.empty((self.model.batch_size, *self.shape), self.model.dtype)
        self.y = TensorGPU(y_gpu, self.model.tensor_format, self.model.cudnn_dtype)

        # Derivative dx
        dx_gpu = gpuarray.empty(self.x.ary.shape, self.model.dtype)
        self.dx = TensorGPU(dx_gpu, self.model.tensor_format, self.model.cudnn_dtype)
        # Derivative dw and derivative db
        if self.model.gpudirect:
            self.dw_cpu = drv.aligned_zeros(self.weights.ary.shape, self.model.dtype)
            self.dw_cpu = dw_gpu = drv.register_host_memory(self.dw_cpu,
                                                            flags=drv.mem_host_register_flags.DEVICEMAP)
            if self.use_bias:
                self.db_cpu = drv.aligned_zeros(self.biases.ary.shape, self.model.dtype)
                self.db_cpu = db_gpu = drv.register_host_memory(self.db_cpu,
                                                                flags=drv.mem_host_register_flags.DEVICEMAP)
        else:
            self.dw_cpu = np.zeros(self.weights.ary.shape, self.model.dtype)
            dw_gpu = gpuarray.empty(self.weights.ary.shape, self.model.dtype)
            if self.use_bias:
                self.db_cpu = np.zeros(self.biases.ary.shape, self.model.dtype)
                db_gpu = gpuarray.empty(self.biases.ary.shape, self.model.dtype)

        self.dw = TensorGPU(dw_gpu, self.model.tensor_format, self.model.cudnn_dtype,
                            tensor_type="filter", gpudirect=self.model.gpudirect)

        if self.use_bias:
            # noinspection PyUnboundLocalVariable
            self.db = TensorGPU(db_gpu, self.model.tensor_format, self.model.cudnn_dtype,
                                gpudirect=self.model.gpudirect)
            
        # Convolution params
        conv_mode = cudnn.cudnnConvolutionMode['CUDNN_CROSS_CORRELATION']
        self.fwd_algo = cudnn.cudnnConvolutionFwdAlgo['CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM']
        self.bwd_dw_algo = cudnn.cudnnConvolutionBwdFilterAlgo['CUDNN_CONVOLUTION_BWD_FILTER_ALGO_1']
        self.bwd_dx_algo = cudnn.cudnnConvolutionBwdDataAlgo['CUDNN_CONVOLUTION_BWD_DATA_ALGO_1']

        # Create convolution descriptor
        self.conv_desc = cudnn.cudnnCreateConvolutionDescriptor()
        cudnn.cudnnSetConvolution2dDescriptor(self.conv_desc, self.vpadding, self.hpadding,
                                              self.vstride, self.hstride, self.vdilation, self.hdilation,
                                              conv_mode, self.model.cudnn_dtype)
        # Set grouping options
        if self.grouping is GroupingEnum.DEPTHWISE:
            cudnn.cudnnSetConvolutionGroupCount(self.conv_desc, self.ci)

        # Allow NCHW -> NHWC conversion for the use of Tensor Cores
        math_type = cudnn.cudnnMathType['CUDNN_TENSOR_OP_MATH_ALLOW_CONVERSION']
        # math_type = cudnn.cudnnMathType['CUDNN_DEFAULT_MATH']
        # math_type = cudnn.cudnnMathType['CUDNN_TENSOR_OP_MATH']
        cudnn.cudnnSetConvolutionMathType(self.conv_desc, math_type)

        # Get output dimensions
        _, _, _ho, _wo = cudnn.cudnnGetConvolution2dForwardOutputDim(self.conv_desc,
                                                                     x.desc, self.weights.desc)
        assert self.ho == _ho and self.wo == _wo, "cuDNN output sizes differ from expected ones!"

        # Set to 20 the number of requested algorithms for enable_cudnn_auto_conv_alg
        req_algs = 20

        self.fwd_algo = cudnn.cudnnFindConvolutionForwardAlgorithm(self.model.cudnn_handle,
                                                                   x.desc, self.weights.desc, self.conv_desc,
                                                                   self.y.desc, req_algs)[0].algo \
            if self.model.enable_cudnn_auto_conv_alg else \
            cudnn.cudnnConvolutionFwdAlgo['CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM']

        local_size = cudnn.cudnnGetConvolutionForwardWorkspaceSize(self.model.cudnn_handle,
                                                                   x.desc, self.weights.desc, self.conv_desc,
                                                                   self.y.desc, self.fwd_algo)
        checkConvolutionMemory(local_size)

        self.bwd_dw_algo = cudnn.cudnnFindConvolutionBackwardFilterAlgorithm(self.model.cudnn_handle,
                                                                             x.desc, self.y.desc, self.conv_desc,
                                                                             self.weights.desc, req_algs)[0].algo \
            if self.model.enable_cudnn_auto_conv_alg else \
            cudnn.cudnnConvolutionBwdFilterAlgo['CUDNN_CONVOLUTION_BWD_FILTER_ALGO_1']

        local_size = cudnn.cudnnGetConvolutionBackwardFilterWorkspaceSize(self.model.cudnn_handle,
                                                                          x.desc, self.y.desc, self.conv_desc,
                                                                          self.weights.desc, self.bwd_dw_algo)
        checkConvolutionMemory(local_size)

        self.bwd_dx_algo = cudnn.cudnnFindConvolutionBackwardDataAlgorithm(self.model.cudnn_handle,
                                                                            self.weights.desc, self.y.desc,
                                                                            self.conv_desc, x.desc,
                                                                            req_algs)[0].algo \
            if self.model.enable_cudnn_auto_conv_alg else \
            cudnn.cudnnConvolutionBwdDataAlgo['CUDNN_CONVOLUTION_BWD_DATA_ALGO_1']

        local_size = cudnn.cudnnGetConvolutionBackwardDataWorkspaceSize(self.model.cudnn_handle,
                                                                        self.weights.desc, self.y.desc,
                                                                        self.conv_desc,
                                                                        x.desc, self.bwd_dx_algo)
        checkConvolutionMemory(local_size)

        self.forward = self._forward_standard
        self.backward = self._backward_standard
    # -----

    def _forward_standard(self, x: TensorGPU) -> TensorGPU:
        alpha, beta = 1.0, 0.0
        # Compute a' = x x weights
        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, self.id * PYDTNN_OPS_EVENTS + PYDTNN_OPS_EVENT_enum.FORWARD_CUDNN)
        cudnn.cudnnConvolutionForward(self.model.cudnn_handle, alpha,
                                      x.desc, x.ptr,
                                      self.weights.desc, self.weights.ptr,
                                      self.conv_desc, self.fwd_algo,
                                      getConvolutionWorkspacePtr(), getConvolutionWorkspaceSize(), beta,
                                      self.y.desc, self.y.ptr)
        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, PYDTNN_EVENT_FINISHED)

        if self.use_bias:
            alpha, beta = 1.0, 1.0
            self.model.tracer.emit_event(PYDTNN_OPS_EVENT,
                                         self.id * PYDTNN_OPS_EVENTS + PYDTNN_OPS_EVENT_enum.FORWARD_CUDNN_SUM_BIASES)
            # Compute a = a' + biases
            cudnn.cudnnAddTensor(self.model.cudnn_handle, alpha, self.biases.desc, self.biases.ptr,
                                 beta, self.y.desc, self.y.ptr)
            self.model.tracer.emit_event(PYDTNN_OPS_EVENT, PYDTNN_EVENT_FINISHED)
        return self.y
    # -----

    def _backward_standard(self, dy: TensorGPU) -> TensorGPU:
        alpha, beta = 1.0, 0.0
        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, self.id * PYDTNN_OPS_EVENTS + PYDTNN_OPS_EVENT_enum.BACKWARD_CUDNN_DW)
        # Compute dw
        cudnn.cudnnConvolutionBackwardFilter(self.model.cudnn_handle, alpha,
                                             self.x.desc, self.x.ptr,
                                             dy.desc, dy.ptr, self.conv_desc, self.bwd_dw_algo,
                                             getConvolutionWorkspacePtr(), getConvolutionWorkspaceSize(), beta,
                                             self.dw.desc, self.dw.ptr)
        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, PYDTNN_EVENT_FINISHED)

        # DtoH dw when data parallelism and no GPU direct/NCCL is used
        if self.model.comm and not self.model.gpudirect and not self.model.enable_nccl:
            self.model.stream.synchronize()
            self.dw.ary.get_async(self.stream_2, self.dw_cpu)

        if self.use_bias:
            self.model.tracer.emit_event(PYDTNN_OPS_EVENT, self.id * PYDTNN_OPS_EVENTS + PYDTNN_OPS_EVENT_enum.BACKWARD_CUDNN_DB)
            # Compute db
            cudnn.cudnnConvolutionBackwardBias(self.model.cudnn_handle, alpha,
                                               dy.desc, dy.ptr, beta,
                                               self.db.desc, self.db.ptr)
            self.model.tracer.emit_event(PYDTNN_OPS_EVENT, PYDTNN_EVENT_FINISHED)

            # DtoH db when data parallelism and no GPU direct/NCCL is used
            if self.model.comm and not self.model.gpudirect and not self.model.enable_nccl:
                self.model.stream.synchronize()
                self.db.ary.get_async(self.stream_2, self.db_cpu)

        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, self.id * PYDTNN_OPS_EVENTS + PYDTNN_OPS_EVENT_enum.BACKWARD_CUDNN_DX)
        # Compute dx
        cudnn.cudnnConvolutionBackwardData(self.model.cudnn_handle, alpha,
                                            self.weights.desc, self.weights.ptr,
                                            dy.desc, dy.ptr,
                                            self.conv_desc, self.bwd_dx_algo,
                                            getConvolutionWorkspacePtr(), getConvolutionWorkspaceSize(), beta,
                                            self.dx.desc, self.dx.ptr)
        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, PYDTNN_EVENT_FINISHED)
        return self.dx
    # ----

    ####################
    ## DEPTHWISE CONV ##
    ####################

    def cuda_depthwise_conv_2d_fwd(self, _func_name: str, _macro_shift_pointer: str) -> Function:

        code = \
"""
{macro_shift_pointer}
#define INDEX_C (idx, c, kh, kw) (idx / (kh * kw)) % c;
#define INDEX_KH (idx, kh, kw) (idx / kw) % kh;
#define INDEX_KW (idx, kw) idx % kw;

__global__ void {func_name}({T}* x, {T}* k, {T}* res, 
                            int vpadding, int hpadding,
                            int vstride, int hstride, 
                            int vdilation, int hdilation, 
                            int n, int c, int h, int w, 
                            int kh, int kw, int ho, int wo
                            int num_workers)
{{
    int idx, cc, ii, jj, yy, xx, nn, x_x, x_y;
    int N = c * kh * kw; 

    for(idx = blockIdx.x * blockDim.x + threadIdx.x; i < N; i += num_workers)
    {{
        cc = INDEX_C(idx, c, kh, kw);
        ii = INDEX_KH(idx, kh, kw);
        jj = INDEX_KW(idx, kw);

        for (nn = 0; nn < n; nn++)
        {{
            for (xx = 0; xx < ho; xx++)
            {{
                x_x = vstride * xx + vdilation * ii - vpadding
                if ((0 <= x_x) && (x_x < h))
                {{
                    for (yy = 0; y < wo; y++)
                    {{
                        x_y = hstride * yy + hdilation * jj - hpadding
                        if ((0 <= x_y) && (x_y < w)) 
                        {{
                            (*SHIFT_POINTER(res, c, h, w, nn, cc, xx, xy)) += ({T}) (*(k+idx) * (*SHIFT_POINTER(x, c, h, w, nn, cc, x_x, x_y)))
                        }}
                    }}
                }}
            }}
        }}   
    }}            

}}
"""
        _t = DICT_SUPPORTED_TYPES[self.model.dtype] # variable Type

        code = code.format(macro_shift_pointer = _macro_shift_pointer,
                           func_name = _func_name,
                           T = _t
                           )
        module = SourceModule(code).get_function(_func_name)
        
        return module
    # ---

    def cuda_depthwise_conv_2d_bwd(self, _func_name: str, _macro_shift_pointer: str) -> Function:

        code = \
"""
{macro_shift_pointer}
#define INDEX_C (idx, c, kh, kw) (idx / (kh * kw)) % c;
#define INDEX_KH (idx, kh, kw) (idx / kw) % kh;
#define INDEX_KW (idx, kw) idx % kw;

__global__ void {func_name}({T}* dy, {T}* x, {T}* k, 
                            {T}* dx, {T}* dw,
                            int vpadding, int hpadding,
                            int vstride, int hstride, 
                            int vdilation, int hdilation, 
                            int n, int c, int h, int w, 
                            int kh, int kw, int ho, int wo
                            int num_workers)
{{
    int idx, cc, ii, jj, yy, xx, nn, x_x, x_y;
    {T}* val_k, val_dy;

    int N = c * kh * kw; 

    for(idx = blockIdx.x * blockDim.x + threadIdx.x; i < N; i += num_workers)
    {{
        val_k = *(k+idx);
        cc = INDEX_C(idx, c, kh, kw);
        ii = INDEX_KH(idx, kh, kw);
        jj = INDEX_KW(idx, kw);

        for (nn = 0; nn < n; nn++)
        {{
            for (xx = 0; xx < ho; xx++)
            {{
                x_x = vstride * xx + vdilation * ii - vpadding;
                if ((0 <= x_x) && (x_x < h)){{
                    for (yy = 0; y < wo; y++)
                    {{
                        x_y = hstride * yy + hdilation * jj - hpadding;
                        val_dy = ({T}) *(SHIFT_POINTER(dy, c, h, w, nn, cc, xx, yy));
                        if ((0 <= x_y) && (x_y < w)) {{
                            *(dw + idx) = ({T}) (*SHIFT_POINTER(x, c, h, w, nn, cc, x_x, x_y)) * val_dy;
                            (*SHIFT_POINTER(dx, c, h, w, nn, cc, x_x, x_y)) += ({T}) (val_k * val_dy);
                        }}
                    }}
                }}
            }}
        }}   
    }}            

}}
"""
        _t = DICT_SUPPORTED_TYPES[self.model.dtype] # variable Type

        code = code.format(macro_shift_pointer = _macro_shift_pointer,
                           func_name = _func_name,
                           T = _t
                           )
        module = SourceModule(code).get_function(_func_name)
        
        return module
    # ---

    def cuda_bias_sum_fwd_depthwise_conv(self, _func_name:str = "bias_sum_fwd_depthwise_conv") -> Function:
        _t = DICT_SUPPORTED_TYPES[self.model.dtype] # variable Type

        code = \
"""
__global__ void {func_name}({T}* x, {T}* bias
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

        code = code.format(func_name = _func_name,
                           T = _t
                           )
        module = SourceModule(code).get_function(_func_name)
        
        return module
    #----

    def cuda_bias_sum_bwd_depthwise_conv_nchw(self, _func_name:str = "bias_sum_bwd_depthwise_conv_nchw") -> Function:
        _t = DICT_SUPPORTED_TYPES[self.model.dtype] # variable Type

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

        code = code.format(func_name = _func_name,
                           T = _t
                           )
        module = SourceModule(code).get_function(_func_name)
        
        return module
    #----

    def cuda_bias_sum_bwd_depthwise_conv_nhwc(self, _func_name:str = "bias_sum_bwd_depthwise_conv_nhwc") -> Function:
        _t = DICT_SUPPORTED_TYPES[self.model.dtype] # variable Type

        # np.sum(dy, axis=(0, 1, 2), out=self.db)
        code = \
"""
__global__ void {func_name}({T}* dy, {T}* db
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

        code = code.format(func_name = _func_name,
                           T = _t
                           )
        module = SourceModule(code).get_function(_func_name)
        
        return module
    #----

    def initialize_depthwise_grouping(self):
        func_name:str = None
        shift_pointer_macro:str = None
        self.bias_sum_bwd:Function = None

        match self.model.tensor_format:
            case PYDTNN_TENSOR_FORMAT.NCHW:
                func_name = "cuda_depthwise_conv_2d_{fwd_bwd}_nchw"
                shift_pointer_macro = "#define SHIFT_POINTER (p, c, h, w, ni, hi, wi) p + ((ni * c + ci) * h + hi) * w + wi"
                self.bias_sum_bwd = self.cuda_bias_sum_bwd_depthwise_conv_nchw()
                self.forward = self._forward_depthwise_nchw
                self.backward = self._backward_depthwise_nchw
            case PYDTNN_TENSOR_FORMAT.NHWC:
                func_name = "cuda_depthwise_conv_2d_{fwd_bwd}_nhwc"
                shift_pointer_macro = "#define SHIFT_POINTER (p, c, h, w, ni, hi, wi) p + ((ni * h + hi) * w + wi) * c + ci"
                self.bias_sum_bwd = self.cuda_bias_sum_bwd_depthwise_conv_nhwc()
                self.forward = self._forward_depthwise_nhwc
                self.backward = self._backward_depthwise_nhwc
            case _:
                raise NotImplementedError(f"\"conv_2d_gpu_depthwise\" is not implemented for \"{self.model.tensor_format}\" format.")

        self.total_num_threads = np.prod(self.grid, dtype=np.int32) * np.prod(self.block, dtype=np.int32)

        self.fwd_func:Function = self.cuda_depthwise_conv_2d_fwd(func_name.format(fwd_bwd="fwd"), shift_pointer_macro)
        self.bwd_func:Function = self.cuda_depthwise_conv_2d_bwd(func_name.format(fwd_bwd="bwd"), shift_pointer_macro)
        self.bias_sum_fwd:Function = self.cuda_bias_sum_fwd_depthwise_conv()

        # Derivative dw and derivative db
        if self.model.gpudirect:
            self.dw_cpu = drv.aligned_zeros(self.weights.ary.shape, self.model.dtype)
            self.dw_cpu = dw_gpu = drv.register_host_memory(self.dw_cpu,
                                                            flags=drv.mem_host_register_flags.DEVICEMAP)
            if self.use_bias:
                self.db_cpu = drv.aligned_zeros(self.biases.ary.shape, self.model.dtype)
                self.db_cpu = db_gpu = drv.register_host_memory(self.db_cpu,
                                                                flags=drv.mem_host_register_flags.DEVICEMAP)
        else:
            self.dw_cpu = np.zeros(self.weights.ary.shape, self.model.dtype)
            dw_gpu = gpuarray.empty(self.weights.ary.shape, self.model.dtype)
            if self.use_bias:
                self.db_cpu = np.zeros(self.biases.ary.shape, self.model.dtype)
                db_gpu = gpuarray.empty(self.biases.ary.shape, self.model.dtype)

        self.dw = TensorGPU(dw_gpu, self.model.tensor_format, self.model.cudnn_dtype,
                            tensor_type="filter", gpudirect=self.model.gpudirect)

        if self.use_bias:
            # noinspection PyUnboundLocalVariable
            self.db = TensorGPU(db_gpu, self.model.tensor_format, self.model.cudnn_dtype,
                                gpudirect=self.model.gpudirect)
    # ---- 

    def _forward_depthwise_nchw(self, x: TensorGPU) -> TensorGPU:
        self.x = x
        y_gpu = gpuarray.to_gpu(np.zeros(shape = (x.shape[0], *self.shape), dtype=self.model.dtype), self.model.dtype)
        self.y = TensorGPU(y_gpu, self.model.tensor_format, self.model.cudnn_dtype)

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
            self.model.tracer.emit_event(PYDTNN_OPS_EVENT, self.id * PYDTNN_OPS_EVENTS + PYDTNN_OPS_EVENT_enum.FORWARD_CUDNN_SUM_BIASES)
            self.bias_sum_fwd(x.ary, self.biases.ary, n * c * h * w,
                          self.total_num_threads, grid=self.grid, block=self.block,
                          stream=self.model.stream)
            self.model.tracer.emit_event(PYDTNN_OPS_EVENT, PYDTNN_EVENT_FINISHED)

        return self.y
    # ----

    def _forward_depthwise_nhwc(self, x: TensorGPU) -> TensorGPU:
        self.x = x
        y_gpu = gpuarray.to_gpu(np.zeros(shape = (x.shape[0], *self.shape), dtype=self.model.dtype), self.model.dtype)
        self.y = TensorGPU(y_gpu, self.model.tensor_format, self.model.cudnn_dtype)

        n, h, w, c = x.shape        

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
            self.model.tracer.emit_event(PYDTNN_OPS_EVENT, self.id * PYDTNN_OPS_EVENTS + PYDTNN_OPS_EVENT_enum.FORWARD_CUDNN_SUM_BIASES)
            self.bias_sum_fwd(x.ary, self.biases.ary, n * h * w * c,
                          self.total_num_threads, grid=self.grid, block=self.block,
                          stream=self.model.stream)
            self.model.tracer.emit_event(PYDTNN_OPS_EVENT, PYDTNN_EVENT_FINISHED)

        return self.y
    # ----

    def _backward_depthwise_nchw(self, dy: TensorGPU) -> TensorGPU:
        dx_gpu = gpuarray.to_gpu(np.zeros(shape = (dy.shape[0], *self.shape), dtype=self.model.dtype), self.model.dtype)
        dx = TensorGPU(dx_gpu, self.model.tensor_format, self.model.cudnn_dtype)

        n, c, h, w = dy.shape

        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, self.id * PYDTNN_OPS_EVENTS + PYDTNN_OPS_EVENT_enum.BACKWARD_CUDNN_DX)
        self.fwd_func(dy.ary, self.x.ary, self.weights.ary,
                      dx.ary, self.dw.ary,
                      np.int32(self.vpadding), np.int32(self.hpadding),
                      np.int32(self.vstride), np.int32(self.hstride),
                      np.int32(self.vdilation), np.int32(self.hdilation),      
                      np.int32(n), np.int32(c), np.int32(h), np.int32(w), 
                      np.int32(self.kh), np.int32(self.kw), np.int32(self.ho), np.int32(self.wo),
                      self.total_num_threads, 
                      grid=self.grid, block=self.block, stream=self.model.stream)
        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, PYDTNN_EVENT_FINISHED)
        
        if self.use_bias:
            self.model.tracer.emit_event(PYDTNN_OPS_EVENT, self.id * PYDTNN_OPS_EVENTS + PYDTNN_OPS_EVENT_enum.BACKWARD_CUDNN_DB)
            self.bias_sum_bwd(dy.ary, self.db.ary,
                              np.int32(c), np.int32(h), np.int32(w), 
                              np.int32(n*c*h*w), self.total_num_threads, 
                              grid=self.grid, block=self.block, stream=self.model.stream)
            self.model.tracer.emit_event(PYDTNN_OPS_EVENT, PYDTNN_EVENT_FINISHED)

        return dx.ary
    # -----

    def _backward_depthwise_nhwc(self, dy: TensorGPU) -> TensorGPU:
        dx_gpu = gpuarray.to_gpu(np.zeros(shape = (dy.shape[0], *self.shape), dtype=self.model.dtype), self.model.dtype)
        dx = TensorGPU(dx_gpu, self.model.tensor_format, self.model.cudnn_dtype)

        n, h, w, c = dy.shape

        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, self.id * PYDTNN_OPS_EVENTS + PYDTNN_OPS_EVENT_enum.BACKWARD_CUDNN_DX)
        self.fwd_func(dy.ary, self.x.ary, self.weights.ary,
                      dx.ary, self.dw.ary,
                      np.int32(self.vpadding), np.int32(self.hpadding),
                      np.int32(self.vstride), np.int32(self.hstride),
                      np.int32(self.vdilation), np.int32(self.hdilation),      
                      np.int32(n), np.int32(c), np.int32(h), np.int32(w), 
                      np.int32(self.kh), np.int32(self.kw), np.int32(self.ho), np.int32(self.wo),
                      self.total_num_threads, grid=self.grid, block=self.block,
                      stream=self.model.stream)
        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, PYDTNN_EVENT_FINISHED)
    
        if self.use_bias:
            self.model.tracer.emit_event(PYDTNN_OPS_EVENT, self.id * PYDTNN_OPS_EVENTS + PYDTNN_OPS_EVENT_enum.BACKWARD_CUDNN_DB)
            self.bias_sum_bwd(dy.ary, self.db.ary,
                              np.int32(c), np.int32(n*h*w*c), 
                              self.total_num_threads, 
                              grid=self.grid, block=self.block, stream=self.model.stream)
            self.model.tracer.emit_event(PYDTNN_OPS_EVENT, PYDTNN_EVENT_FINISHED)

        return dx.ary
    # -----

    ####################
    ## POINTWISE CONV ##
    ####################

    def initialize_pointwise_grouping(self):
        pass

    def _forward_pointwise_nchw(self, x: TensorGPU) -> TensorGPU:
        pass

    def _backward_pointwise_nchw(self, dy: TensorGPU) -> TensorGPU:
        pass

    def _forward_pointwise_nhwc(self, x: TensorGPU) -> TensorGPU:
        pass

    def _backward_pointwise_nhwc(self, dy: TensorGPU) -> TensorGPU:
        pass

