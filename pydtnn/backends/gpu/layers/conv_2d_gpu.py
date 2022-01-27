#
#  This file is part of Python Distributed Training of Neural Networks (PyDTNN)
#
#  Copyright (C) 2021-22 Universitat Jaume I
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

from pydtnn.performance_models import *
from pydtnn.tracers import PYDTNN_OPS_EVENT, PYDTNN_OPS_EVENTS, PYDTNN_OPS_FORWARD_CUDNN, \
    PYDTNN_OPS_FORWARD_CUDNN_SUM_BIASES, \
    PYDTNN_OPS_BACKWARD_CUDNN_DW, PYDTNN_OPS_BACKWARD_CUDNN_DB, PYDTNN_OPS_BACKWARD_CUDNN_DX
from .layer_gpu import LayerGPU
from .memory_allocation import checkConvolutionMemory, getConvolutionWorkspaceSize, getConvolutionWorkspacePtr
from ..tensor_gpu import TensorGPU
from pydtnn.utils import PYDTNN_TENSOR_FORMAT_NHWC, PYDTNN_TENSOR_FORMAT_NCHW

class Conv2DGPU(LayerGPU, Conv2D):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.fwd_algo = None
        self.fwd_time = None
        self.bwd_dw_algo = None
        self.bwd_dx_algo = None
        self.conv_desc = None

    def initialize(self, prev_shape, need_dx, x):
        super().initialize(prev_shape, need_dx, x)
        # This weight shape is required for cuDNN when NHWC is seleted!
        if self.model.tensor_format == PYDTNN_TENSOR_FORMAT_NHWC:
            self.weights_shape = (self.co, *self.filter_shape, self.ci)

        self.stream_2 = drv.Stream()
        # Convolution params
        conv_mode = cudnn.cudnnConvolutionMode['CUDNN_CROSS_CORRELATION']
        self.fwd_algo = cudnn.cudnnConvolutionFwdAlgo['CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM']
        self.bwd_dw_algo = cudnn.cudnnConvolutionBwdFilterAlgo['CUDNN_CONVOLUTION_BWD_FILTER_ALGO_1']
        self.bwd_dx_algo = cudnn.cudnnConvolutionBwdDataAlgo['CUDNN_CONVOLUTION_BWD_DATA_ALGO_1']

        self.weights_cpu = self.weights_initializer(self.weights_shape, self.model.dtype)
        weights_gpu = gpuarray.to_gpu(self.weights_cpu)
        self.weights = TensorGPU(weights_gpu, self.model.tensor_format, self.model.cudnn_dtype, "filter")
        # Biases
        if self.use_bias:
            self.biases_cpu = self.biases_initializer((1, self.co, 1, 1) \
               if self.model.tensor_format == PYDTNN_TENSOR_FORMAT_NCHW else (1, 1, 1, self.co), self.model.dtype)
            biases_gpu = gpuarray.to_gpu(self.biases_cpu)
            self.biases = TensorGPU(biases_gpu, self.model.tensor_format, self.model.cudnn_dtype)
        # Create convolution descriptor
        self.conv_desc = cudnn.cudnnCreateConvolutionDescriptor()
        cudnn.cudnnSetConvolution2dDescriptor(self.conv_desc, self.vpadding, self.hpadding,
                                              self.vstride, self.hstride, self.vdilation, self.hdilation,
                                              conv_mode, self.model.cudnn_dtype)
        # Set grouping options
        if self.grouping == "depthwise":
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

        # Activations y
        y_gpu = gpuarray.empty((self.model.batch_size, *self.shape), self.model.dtype)
        self.y = TensorGPU(y_gpu, self.model.tensor_format, self.model.cudnn_dtype)

        # Derivative dx
        if self.need_dx:
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

        if self.need_dx:
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

        self.fwd_time = \
            matmul_time(m=self.co, n=(self.model.batch_size * self.ho * self.wo), k=(self.ci * self.kh * self.kw),
                        cpu_speed=self.model.cpu_speed, memory_bw=self.model.memory_bw, dtype=self.model.dtype)
        self.bwd_time = \
            matmul_time(m=self.co, n=(self.ci * self.kh * self.kw), k=(self.model.batch_size * self.ho * self.wo),
                        cpu_speed=self.model.cpu_speed, memory_bw=self.model.memory_bw, dtype=self.model.dtype) + \
            matmul_time(m=(self.ci * self.kh * self.kw), n=(self.model.batch_size * self.ho * self.wo), k=self.co,
                        cpu_speed=self.model.cpu_speed, memory_bw=self.model.memory_bw, dtype=self.model.dtype) \
            if need_dx else 0

    def forward(self, x):
        alpha, beta = 1.0, 0.0
        # Compute a' = x x weights
        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, self.id * PYDTNN_OPS_EVENTS + PYDTNN_OPS_FORWARD_CUDNN)
        cudnn.cudnnConvolutionForward(self.model.cudnn_handle, alpha,
                                      x.desc, x.ptr,
                                      self.weights.desc, self.weights.ptr,
                                      self.conv_desc, self.fwd_algo,
                                      getConvolutionWorkspacePtr(), getConvolutionWorkspaceSize(), beta,
                                      self.y.desc, self.y.ptr)
        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, 0)

        if self.use_bias:
            alpha, beta = 1.0, 1.0
            self.model.tracer.emit_event(PYDTNN_OPS_EVENT,
                                         self.id * PYDTNN_OPS_EVENTS + PYDTNN_OPS_FORWARD_CUDNN_SUM_BIASES)
            # Compute a = a' + biases
            cudnn.cudnnAddTensor(self.model.cudnn_handle, alpha, self.biases.desc, self.biases.ptr,
                                 beta, self.y.desc, self.y.ptr)
            self.model.tracer.emit_event(PYDTNN_OPS_EVENT, 0)
        return self.y

    def backward(self, dy):
        alpha, beta = 1.0, 0.0
        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, self.id * PYDTNN_OPS_EVENTS + PYDTNN_OPS_BACKWARD_CUDNN_DW)
        # Compute dw
        cudnn.cudnnConvolutionBackwardFilter(self.model.cudnn_handle, alpha,
                                             self.x.desc, self.x.ptr,
                                             dy.desc, dy.ptr, self.conv_desc, self.bwd_dw_algo,
                                             getConvolutionWorkspacePtr(), getConvolutionWorkspaceSize(), beta,
                                             self.dw.desc, self.dw.ptr)
        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, 0)

        # DtoH dw when data parallelism and no GPU direct/NCCL is used
        if self.model.comm and not self.model.gpudirect and not self.model.enable_nccl:
            self.model.stream.synchronize()
            self.dw.ary.get_async(self.stream_2, self.dw_cpu)

        if self.use_bias:
            self.model.tracer.emit_event(PYDTNN_OPS_EVENT, self.id * PYDTNN_OPS_EVENTS + PYDTNN_OPS_BACKWARD_CUDNN_DB)
            # Compute db
            cudnn.cudnnConvolutionBackwardBias(self.model.cudnn_handle, alpha,
                                               dy.desc, dy.ptr, beta,
                                               self.db.desc, self.db.ptr)
            self.model.tracer.emit_event(PYDTNN_OPS_EVENT, 0)

            # DtoH db when data parallelism and no GPU direct/NCCL is used
            if self.model.comm and not self.model.gpudirect and not self.model.enable_nccl:
                self.model.stream.synchronize()
                self.db.ary.get_async(self.stream_2, self.db_cpu)

        if self.need_dx:
            self.model.tracer.emit_event(PYDTNN_OPS_EVENT, self.id * PYDTNN_OPS_EVENTS + PYDTNN_OPS_BACKWARD_CUDNN_DX)
            # Compute dx
            cudnn.cudnnConvolutionBackwardData(self.model.cudnn_handle, alpha,
                                               self.weights.desc, self.weights.ptr,
                                               dy.desc, dy.ptr,
                                               self.conv_desc, self.bwd_dx_algo,
                                               getConvolutionWorkspacePtr(), getConvolutionWorkspaceSize(), beta,
                                               self.dx.desc, self.dx.ptr)
            self.model.tracer.emit_event(PYDTNN_OPS_EVENT, 0)
            return self.dx
