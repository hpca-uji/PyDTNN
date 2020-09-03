""" Python Distributed Training of Neural Networks - PyDTNN

PyDTNN is a light-weight library for distributed Deep Learning training and 
inference that offers an initial starting point for interaction with 
distributed training of (and inference with) deep neural networks. PyDTNN 
priorizes simplicity over efficiency, providing an amiable user interface 
which enables a flat accessing curve. To perform the training and inference 
processes, PyDTNN exploits distributed inter-process parallelism (via MPI) 
for clusters and intra-process (via multi-threading) parallelism to leverage 
the presence of multicore processors and GPUs at node level. For that, PyDTNN 
uses MPI4Py for message-passing, BLAS calls via NumPy for multicore processors
and PyCUDA+cuDNN+cuBLAS for NVIDIA GPUs.

This program is free software: you can redistribute it and/or modify it under
the terms of the GNU General Public License as published by the Free Software
Foundation, either version 3 of the License, or (at your option) any later
version.

This program is distributed in the hope that it will be useful, but WITHOUT
ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
You should have received a copy of the GNU General Public License along with
this program. If not, see <http://www.gnu.org/licenses/>.

"""

__author__ = "Manuel F. Dolz, Enrique S. Quintana, \
              Mar Catalan, Adrian Castello"
__contact__ = "dolzm@uji.es"
__copyright__ = "Copyright 2020, Universitat Jaume I"
__credits__ = ["Manuel F. Dolz, Enrique S. Quintana", \
               "Mar Catalan", "Adrian Castello"]
__date__ = "2020/03/22"

__email__ =  "dolzm@uji.es"
__license__ = "GPLv3"
__maintainer__ = "Manuel F. Dolz, Adrian Castello (GPU)"
__status__ = "Production"
__version__ = "1.1.0"


import numpy as np
import NN_util, NN_activation, NN_initializer, NN_layer

from math import floor
from NN_util import printf, TensorGPU
from NN_im2col_cython import im2col_cython, col2im_cython
from NN_argmax_cython import argmax_cython
from NN_add_cython import add_cython
from NN_tracer import PYDL_EVT, PYDL_OPS_EVT, PYDL_NUM_EVTS, PYDL_OPS_EVT, PYDL_OPS_NUM_EVTS
from NN_sim import *

try:
    from mpi4py import MPI
    import pycuda.gpuarray as gpuarray
    import pycuda.driver as drv
    from pycuda.elementwise import ElementwiseKernel
    from skcuda import cublas
    import libcudnn.libcudnn as cudnn
    import ctypes

except:
    pass

# The Code below will allocate the maximum used memory
# and that memory will be shared among all layers.
# This code saves having a memory allocation per layer
ws_size = 1 
ws = drv.mem_alloc(ws_size) if ws_size > 0 else 0
ws_ptr = ctypes.c_void_p(int(ws))

def checkConvolutionMemory(size):
    global ws_size
    global ws
    global ws_ptr
    # if a layer requires more memory than the allocated
    # we re-allocated that size
    if size.value > ws_size:
        ws_size = size.value
        ws.free()
        ws = drv.mem_alloc(ws_size) if ws_size > 0 else 0
        ws_ptr = ctypes.c_void_p(int(ws))


class InputGPU(NN_layer.Input):
    
    def initialize(self, prev_shape, need_dx, x):
        y_gpu = gpuarray.empty((self.batch_size, *self.shape), self.dtype)
        self.y = TensorGPU(y_gpu, self.tensor_fmt, self.cudnn_dtype)


class FCGPU(NN_layer.FC):

    def initialize(self, prev_shape, need_dx, x):
        self.need_dx = need_dx
        self.x = x
        self.stream_2 = drv.Stream()

        # Weights
        self.weights_cpu = self.weights_initializer((*prev_shape, *self.shape), self.dtype)
        weights_gpu = gpuarray.to_gpu(self.weights_cpu)
        self.weights = TensorGPU(weights_gpu, self.tensor_fmt, self.cudnn_dtype)

        if self.use_bias:
            # Biases
            self.biases_cpu = self.biases_initializer((1, *self.shape), self.dtype)
            biases_gpu  = gpuarray.to_gpu(self.biases_cpu)
            self.biases = TensorGPU(biases_gpu, self.tensor_fmt, self.cudnn_dtype)

        y_gpu = gpuarray.empty((self.batch_size, self.shape[0]), self.dtype)
        self.y = TensorGPU(y_gpu, self.tensor_fmt, self.cudnn_dtype)

        if self.need_dx:
            dx_gpu = gpuarray.empty(self.x.ary.shape, self.dtype)
            self.dx = TensorGPU(dx_gpu, self.tensor_fmt, self.cudnn_dtype)
        
        self.dw_cpu = np.zeros(self.weights.ary.shape, self.dtype)
        if self.use_bias: self.db_cpu = np.zeros(self.biases.ary.shape, self.dtype)

        if self.gpudirect:
            dw_gpu = drv.register_host_memory(self.dw_cpu)
            if self.use_bias: db_gpu = drv.register_host_memory(self.db_cpu)
        else:
            dw_gpu = gpuarray.empty(self.dw_cpu.shape, self.dtype)
            if self.use_bias: db_gpu = gpuarray.empty(self.db_cpu.shape, self.dtype)

        self.dw = TensorGPU(dw_gpu, self.tensor_fmt, self.cudnn_dtype, gpudirect=self.gpudirect)
        if self.use_bias: 
            self.db = TensorGPU(db_gpu, self.tensor_fmt, self.cudnn_dtype, gpudirect=self.gpudirect)

        self.onevec_gpu = gpuarray.to_gpu(np.ones((self.batch_size), self.dtype))
        self.nparams = self.weights.size + (self.biases.size if self.use_bias else 0)
        self.matmul, self.matvec = NN_util.matmul_gpu, NN_util.matvec_gpu

        self.fwd_time = \
            matmul_time(m=self.batch_size, n=self.weights_cpu.shape[1], k=self.weights_cpu.shape[0], 
                        cpu_speed=self.model.params.cpu_speed, memory_bw=self.model.params.memory_bw, 
                        dtype=self.dtype)
        self.bwd_time = \
            matmul_time(m=self.weights_cpu.shape[0], n=self.weights_cpu.shape[1], k=self.batch_size, 
                        cpu_speed=self.model.params.cpu_speed, memory_bw=self.model.params.memory_bw, 
                        dtype=self.dtype) + \
            matmul_time(m=self.batch_size, n=self.weights_cpu.shape[0], k=self.weights_cpu.shape[1], 
                        cpu_speed=self.model.params.cpu_speed, memory_bw=self.model.params.memory_bw, 
                        dtype=self.dtype) if need_dx else 0

    def forward(self, x):
        m = x.ary.shape[0]
        n = ldb = ldc = self.weights.ary.shape[1]
        k = lda = x.ary.shape[1]
        transA, transB, alpha, beta = 'N', 'N', 1.0, 0.0
        
        # Compute a' = x @ weights 
        self.matmul(self.cublas_handle, transB, transA, n, m, k, alpha, 
                    self.weights.ary.gpudata, ldb, 
                    x.ary.gpudata, lda, beta, 
                    self.y.ary.gpudata, ldc, self.dtype)

        if self.use_bias:
            alpha, beta = 1.0, 1.0
            # Compute a = a' + biases
            cudnn.cudnnAddTensor(self.cudnn_handle, alpha, self.biases.desc, 
                                 self.biases.ptr, beta, self.y.desc, self.y.ptr)
        return self.y

    def backward(self, prev_dx):
        # Compute dw
        m = lda = self.x.ary.shape[1]
        n = ldb = ldc = prev_dx.ary.shape[1]
        k = prev_dx.ary.shape[0]
        transA, transB, alpha, beta = 'T', 'N', 1.0, 0.0

        self.matmul(self.cublas_handle, transB, transA, n, m, k, alpha,
                    prev_dx.ary.gpudata, ldb, self.x.ary.gpudata, lda, beta, 
                    self.dw.ptr_intp if self.gpudirect else self.dw.ary.gpudata, ldc, self.dtype)

        # DtoH dw when data parallelism and no GPU direct is used
        if self.model.comm and not self.gpudirect:
            self.dw.ary.get_async(self.stream_2, self.dw_cpu)
        
        if self.use_bias:
            # Compute db
            m = prev_dx.ary.shape[0]
            n = lda = prev_dx.ary.shape[1]
            transA, alpha, beta, incx, incy = 'N', 1.0, 0.0, 1, 1
    
            self.matvec(self.cublas_handle, transA, n, m, alpha, 
                        prev_dx.ary.gpudata, lda, self.onevec_gpu.gpudata, incx, beta, 
                        self.db.ptr_intp if self.gpudirect else self.db.ary.gpudata, 
                        incy, self.dtype)
    
            # DtoH db when data parallelism and no GPU direct is used
            if self.model.comm and not self.gpudirect:
                self.db.ary.get_async(self.stream_2, self.db_cpu)
            
        if self.need_dx:
            # Compute dx
            m = prev_dx.ary.shape[0]
            n = ldc = self.weights.ary.shape[0]
            k = lda = ldb = prev_dx.ary.shape[1]
            transA, transB, alpha, beta = 'N', 'T', 1.0, 0.0

            self.matmul(self.cublas_handle, transB, transA, n, m, k, alpha,
                        self.weights.ary.gpudata, ldb, 
                        prev_dx.ary.gpudata, lda, beta, 
                        self.dx.ary.gpudata, ldc, self.dtype)
            return self.dx
        

class Conv2DGPU(NN_layer.Conv2D):

    def initialize(self, prev_shape, need_dx, x):
        self.need_dx = need_dx
        self.x = x
        self.stream_2 = drv.Stream()
        self.ci, self.hi, self.wi = prev_shape
        self.kh, self.kw = self.filter_shape

        # Convolution params
        conv_mode        = cudnn.cudnnConvolutionMode\
                                  ['CUDNN_CROSS_CORRELATION']
        self.fwd_algo    = cudnn.cudnnConvolutionFwdAlgo\
                                  ['CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM']
        self.bwd_dw_algo = cudnn.cudnnConvolutionBwdFilterAlgo\
                                  ['CUDNN_CONVOLUTION_BWD_FILTER_ALGO_1']
        self.bwd_dx_algo = cudnn.cudnnConvolutionBwdDataAlgo\
                                  ['CUDNN_CONVOLUTION_BWD_DATA_ALGO_1']

        # Filters
        self.weights_cpu = self.weights_initializer((self.co, self.ci, *self.filter_shape), self.dtype)
        weights_gpu = gpuarray.to_gpu(self.weights_cpu)
        self.weights = TensorGPU(weights_gpu, self.tensor_fmt, self.cudnn_dtype, "filter")

        if self.use_bias:
            # Biases
            self.biases_cpu = self.biases_initializer((1, self.co), self.dtype)
            biases_gpu = gpuarray.to_gpu(self.biases_cpu)
            self.biases = TensorGPU(biases_gpu, self.tensor_fmt, self.cudnn_dtype)

        # Create convolution descriptor
        self.conv_desc = cudnn.cudnnCreateConvolutionDescriptor()
        upscalex, upscaley = 1, 1
        cudnn.cudnnSetConvolution2dDescriptor(self.conv_desc, self.vpadding, self.hpadding,
                                              self.vstride, self.hstride, upscalex, upscaley, 
                                              conv_mode, self.cudnn_dtype)
        # Get output dimensions  
        _, _, self.ho, self.wo = cudnn.cudnnGetConvolution2dForwardOutputDim(self.conv_desc, 
                                              x.desc, self.weights.desc)
        self.shape = (self.co, self.ho, self.wo)
        self.nparams = self.weights.size + (self.biases.size if self.use_bias else 0)

        # Activations y
        y_gpu = gpuarray.empty((self.batch_size, *self.shape), self.dtype)
        self.y = TensorGPU(y_gpu, self.tensor_fmt, self.cudnn_dtype)

        if self.need_dx:
            # Derivative dx
            dx_gpu = gpuarray.empty(self.x.ary.shape, self.dtype)
            self.dx = TensorGPU(dx_gpu, self.tensor_fmt, self.cudnn_dtype)

        self.dw_cpu = np.zeros(self.weights.ary.shape, self.dtype)
        if self.use_bias: self.db_cpu = np.zeros(self.biases.ary.shape, self.dtype)

        if self.gpudirect:
            dw_gpu = drv.register_host_memory(self.dw_cpu)
            if self.use_bias: db_gpu = drv.register_host_memory(self.db_cpu)
        else:
            dw_gpu = gpuarray.empty(self.weights.ary.shape, self.dtype)
            if self.use_bias: db_gpu = gpuarray.empty(self.biases.ary.shape, self.dtype)

        self.dw = TensorGPU(dw_gpu, self.tensor_fmt, self.cudnn_dtype, 
                            tensor_type="filter", gpudirect=self.gpudirect)
        if self.use_bias:
            self.db = TensorGPU(db_gpu, self.tensor_fmt, self.cudnn_dtype, gpudirect=self.gpudirect)

        local_size = cudnn.cudnnGetConvolutionForwardWorkspaceSize(self.cudnn_handle, 
                                              x.desc, self.weights.desc, self.conv_desc, 
                                              self.y.desc, self.fwd_algo)
        checkConvolutionMemory(local_size)

        local_size = cudnn.cudnnGetConvolutionBackwardFilterWorkspaceSize(self.cudnn_handle,
                                              x.desc, self.y.desc, self.conv_desc, 
                                              self.weights.desc, self.bwd_dw_algo)
        checkConvolutionMemory(local_size)

        if self.need_dx:
            local_size = cudnn.cudnnGetConvolutionBackwardDataWorkspaceSize(self.cudnn_handle,
                                              self.weights.desc, self.y.desc, self.conv_desc,
                                              x.desc, self.bwd_dx_algo)
            checkConvolutionMemory(local_size)

        self.fwd_time = \
            matmul_time(m=self.co, n=(self.batch_size * self.ho * self.wo), k=(self.ci * self.kh * self.kw),
                        cpu_speed=self.model.params.cpu_speed, memory_bw=self.model.params.memory_bw, 
                        dtype=self.dtype) 
        self.bwd_time = \
            matmul_time(m=self.co, n=(self.ci * self.kh * self.kw), k=(self.batch_size * self.ho * self.wo),
                        cpu_speed=self.model.params.cpu_speed, memory_bw=self.model.params.memory_bw, 
                        dtype=self.dtype) + \
            matmul_time(m=(self.ci * self.kh * self.kw), n=(self.batch_size * self.ho * self.wo), k=self.co, 
                        cpu_speed=self.model.params.cpu_speed, memory_bw=self.model.params.memory_bw, 
                        dtype=self.dtype) if need_dx else 0

    def forward(self, x):
        alpha, beta = 1.0, 0.0
        # Compute a' = x x weights
        cudnn.cudnnConvolutionForward(self.cudnn_handle, alpha, 
                                      x.desc, x.ptr, 
                                      self.weights.desc, self.weights.ptr, 
                                      self.conv_desc, self.fwd_algo, ws_ptr, ws_size, beta, 
                                      self.y.desc, self.y.ptr)
        if self.use_bias:
            alpha, beta = 1.0, 1.0
            # Compute a = a' + biases
            cudnn.cudnnAddTensor(self.cudnn_handle, alpha, self.biases.desc, self.biases.ptr, 
                                      beta, self.y.desc, self.y.ptr)
        return self.y
       
    def backward(self, prev_dx):
        alpha, beta = 1.0, 0.0
        # Compute dw    
        cudnn.cudnnConvolutionBackwardFilter(self.cudnn_handle, alpha, 
                                      self.x.desc, self.x.ptr, 
                                      prev_dx.desc, prev_dx.ptr, self.conv_desc, 
                                      self.bwd_dw_algo, ws_ptr, ws_size, beta,
                                      self.dw.desc, self.dw.ptr)

        if self.model.comm and not self.gpudirect:
            self.dw.ary.get_async(self.stream_2, self.dw_cpu)

        if self.use_bias:
            # Compute db
            cudnn.cudnnConvolutionBackwardBias(self.cudnn_handle, alpha, 
                                      prev_dx.desc, prev_dx.ptr, beta, 
                                      self.db.desc, self.db.ptr)
            
            if self.model.comm != None and not self.gpudirect:
                self.db.ary.get_async(self.stream_2, self.db_cpu)
        
        if self.need_dx:
            # Compute dx
            cudnn.cudnnConvolutionBackwardData(self.cudnn_handle, alpha, 
                                      self.weights.desc, self.weights.ptr, 
                                      prev_dx.desc, prev_dx.ptr,
                                      self.conv_desc, self.bwd_dx_algo, ws_ptr, ws_size, 
                                      beta, self.dx.desc, self.dx.ptr)
            return self.dx


class MaxPool2DGPU(NN_layer.MaxPool2D):

    def initialize(self, prev_shape, need_dx, x):
        self.need_dx = need_dx
        self.x = x
        self.ci, self.hi, self.wi = prev_shape
        if self.pool_shape[0] == 0: self.pool_shape = (self.hi, self.pool_shape[1])
        if self.pool_shape[1] == 0: self.pool_shape = (self.pool_shape[0], self.wi)
        self.kh, self.kw = self.pool_shape
        self.co = self.ci
        
        pool_mode = cudnn.cudnnPoolingMode['CUDNN_POOLING_MAX']
        nan_prop = cudnn.cudnnNanPropagation['CUDNN_NOT_PROPAGATE_NAN']
        
        self.pool_desc = cudnn.cudnnCreatePoolingDescriptor()
        cudnn.cudnnSetPooling2dDescriptor(self.pool_desc, pool_mode, nan_prop, 
                                          self.kh, self.kw, self.vpadding, self.hpadding, 
                                          self.vstride, self.hstride)
        # Get output dimensions  
        _, _, self.ho, self.wo = cudnn.cudnnGetPooling2dForwardOutputDim(self.pool_desc, 
                                                                         x.desc)
        self.shape = (self.co, self.ho, self.wo)

        # Activations y
        y_gpu = gpuarray.empty((self.batch_size, *self.shape), self.dtype)
        self.y = TensorGPU(y_gpu, self.tensor_fmt, self.cudnn_dtype)
        
        if self.need_dx:
            # Derivative dx
            dx_gpu = gpuarray.empty(self.x.ary.shape, self.dtype)
            self.dx = TensorGPU(dx_gpu, self.tensor_fmt, self.cudnn_dtype)

        self.fwd_time = \
            im2col_time(m=(self.kh * self.kw), n=(self.batch_size * self.ho * self.wo * self.ci),
                        cpu_speed=self.model.params.cpu_speed, memory_bw=self.model.params.memory_bw, 
                        dtype=self.dtype)
        self.bwd_time = \
            col2im_time(m=(self.kh * self.kw), n=(self.batch_size * self.ho * self.wo * self.ci), 
                        cpu_speed=self.model.params.cpu_speed, memory_bw=self.model.params.memory_bw, 
                        dtype=self.dtype) if need_dx else 0

    def forward(self, x):
        alpha, beta = 1.0, 0.0
        cudnn.cudnnPoolingForward(self.cudnn_handle, self.pool_desc, alpha, 
                                  x.desc, x.ptr, beta, 
                                  self.y.desc, self.y.ptr) 
        return self.y

    def backward(self, prev_dx):
        if self.need_dx:
            alpha, beta = 1.0, 0.0
            # Compute dx
            cudnn.cudnnPoolingBackward(self.cudnn_handle, self.pool_desc, alpha, 
                                       self.y.desc, self.y.ptr, 
                                       prev_dx.desc, prev_dx.ptr, 
                                       self.x.desc, self.x.ptr, 
                                       beta, self.dx.desc, self.dx.ptr)
            return self.dx


class AveragePool2DGPU(NN_layer.AveragePool2D):

    def initialize(self, prev_shape, need_dx, x):
        self.need_dx = need_dx
        self.x = x
        self.ci, self.hi, self.wi = prev_shape
        if self.pool_shape[0] == 0: self.pool_shape = (self.hi, self.pool_shape[1])
        if self.pool_shape[1] == 0: self.pool_shape = (self.pool_shape[0], self.wi)
        self.kh, self.kw = self.pool_shape
        self.co = self.ci

        pool_mode = cudnn.cudnnPoolingMode['CUDNN_POOLING_AVERAGE_COUNT_INCLUDE_PADDING']
        nan_prop = cudnn.cudnnNanPropagation['CUDNN_NOT_PROPAGATE_NAN']
        
        self.pool_desc = cudnn.cudnnCreatePoolingDescriptor()
        cudnn.cudnnSetPooling2dDescriptor(self.pool_desc, pool_mode, nan_prop, 
                                          self.kh, self.kw, self.vpadding, self.hpadding, 
                                          self.vstride, self.hstride)
        # Get output dimensions  
        _, _, self.ho, self.wo = cudnn.cudnnGetPooling2dForwardOutputDim(self.pool_desc, 
                                                                         x.desc)
        self.shape = (self.co, self.ho, self.wo)

        # Activations y
        y_gpu = gpuarray.empty((self.batch_size, *self.shape), self.dtype)
        self.y = TensorGPU(y_gpu, self.tensor_fmt, self.cudnn_dtype)
        
        if self.need_dx:
            # Derivative dx
            dx_gpu = gpuarray.empty(self.x.ary.shape, self.dtype)
            self.dx = TensorGPU(dx_gpu, self.tensor_fmt, self.cudnn_dtype)

        self.fwd_time = \
            im2col_time(m=(self.kh * self.kw), n=(self.batch_size * self.ho * self.wo * self.ci),
                        cpu_speed=self.model.params.cpu_speed, memory_bw=self.model.params.memory_bw, 
                        dtype=self.dtype)
        self.bwd_time = \
            col2im_time(m=(self.kh * self.kw), n=(self.batch_size * self.ho * self.wo * self.ci), 
                        cpu_speed=self.model.params.cpu_speed, memory_bw=self.model.params.memory_bw, 
                        dtype=self.dtype) if need_dx else 0

    def forward(self, x):
        alpha, beta = 1.0, 0.0
        cudnn.cudnnPoolingForward(self.cudnn_handle, self.pool_desc, alpha, 
                                  x.desc, x.ptr, beta, 
                                  self.y.desc, self.y.ptr) 
        return self.y

    def backward(self, prev_dx):
        if self.need_dx:
            alpha, beta = 1.0, 0.0
            # Compute dx
            cudnn.cudnnPoolingBackward(self.cudnn_handle, self.pool_desc, alpha, 
                                       self.y.desc, self.y.ptr, 
                                       prev_dx.desc, prev_dx.ptr, 
                                       self.x.desc, self.x.ptr, 
                                       beta, self.dx.desc, self.dx.ptr)
            return self.dx


class DropoutGPU(NN_layer.Dropout):

    def initialize(self, prev_shape, need_dx, x):
        self.need_dx = need_dx
        self.shape = prev_shape

        # Activations y
        y_gpu = gpuarray.empty((self.batch_size, *self.shape), self.dtype)
        self.y = TensorGPU(y_gpu, self.tensor_fmt, self.cudnn_dtype)
        
        if need_dx:
            # Derivative dx
            dx_gpu = gpuarray.empty((self.batch_size, *self.shape), self.dtype)
            self.dx = TensorGPU(dx_gpu, self.tensor_fmt, self.cudnn_dtype)

        self.states_size = cudnn.cudnnDropoutGetStatesSize(self.cudnn_handle)
        self.space_size = cudnn.cudnnDropoutGetReserveSpaceSize(self.y.desc)

        space_gpu = gpuarray.empty((self.space_size.value,), np.byte)
        self.space = TensorGPU(space_gpu, self.tensor_fmt, self.cudnn_dtype, "other")

        states_gpu = gpuarray.empty((self.states_size.value,), np.byte)
        self.states = TensorGPU(states_gpu, self.tensor_fmt, self.cudnn_dtype, "other")

        self.drop_desc = cudnn.cudnnCreateDropoutDescriptor()
        
        cudnn.cudnnSetDropoutDescriptor(self.drop_desc, self.cudnn_handle, self.rate,
                                        self.states.ptr, self.states_size.value, seed=0)

    def forward(self, x):
        cudnn.cudnnDropoutForward(self.cudnn_handle, self.drop_desc, 
                                  x.desc, x.ptr, 
                                  self.y.desc, self.y.ptr, 
                                  self.space.ptr, self.space_size.value)
        return self.y

    def backward(self, prev_dx):
        if self.need_dx:
            # Compute dx
            cudnn.cudnnDropoutBackward(self.cudnn_handle, self.drop_desc, 
                                       prev_dx.desc, prev_dx.ptr, 
                                       self.dx.desc, self.dx.ptr,
                                       self.space.ptr, self.space_size.value)
            return self.dx


class FlattenGPU(NN_layer.Flatten):

    def initialize(self, prev_shape, need_dx, x):
        self.need_dx = need_dx
        self.shape = (np.prod(prev_shape),)
        self.copy = ElementwiseKernel("T *dst, T *src".replace("T", 
                            {"float32": "float", "float64": "double"}[self.dtype]),
                            "dst[i] = src[i];", "copy")
 
        # Activations y
        y_gpu = gpuarray.empty((self.batch_size, np.prod(prev_shape)), self.dtype)
        self.y = TensorGPU(y_gpu, self.tensor_fmt, self.cudnn_dtype)

        if self.need_dx:
            # Derivative dx
            dx_gpu = gpuarray.empty((self.batch_size, *prev_shape), self.dtype)
            self.dx = TensorGPU(dx_gpu, self.tensor_fmt, self.cudnn_dtype)

    def forward(self, x):
        self.copy(self.y.ary, x.ary, stream=self.stream)
        return self.y

    def backward(self, prev_dx):
        if self.need_dx:
            # Compute dx
            self.copy(self.dx.ary, prev_dx.ary, stream=self.stream)
            return self.dx


class BatchNormalizationGPU(NN_layer.BatchNormalization):

    def initialize(self, prev_shape, need_dx, x):
        self.shape = shape_ = prev_shape
        self.need_dx = need_dx
        self.x = x
        self.stream_2 = drv.Stream()

        # Activations y
        y_gpu = gpuarray.empty(x.ary.shape, self.dtype)
        self.y = TensorGPU(y_gpu, self.tensor_fmt, self.cudnn_dtype)
        
        # Derivative dx
        dx_gpu = gpuarray.zeros(x.ary.shape, self.dtype)
        self.dx = TensorGPU(dx_gpu, self.tensor_fmt, self.cudnn_dtype)

        self.spatial = len(self.shape) > 2
        self.mode = cudnn.cudnnBatchNormMode['CUDNN_BATCHNORM_SPATIAL' if self.spatial else \
                                             'CUDNN_BATCHNORM_PER_ACTIVATION']

        self.gamma_beta_mean_var_desc = cudnn.cudnnCreateTensorDescriptor()
        cudnn.cudnnDeriveBNTensorDescriptor(self.gamma_beta_mean_var_desc, 
                                            x.desc, self.mode)
        if self.spatial:
            shape_ = (1, self.shape[0], 1, 1) # 1 x C x 1 x 1
        else:
            shape_ = (1, self.shape[0], 
                         self.shape[1] if len(self.shape) > 2 else 1,
                         self.shape[2] if len(self.shape) > 3 else 1)  # 1 x C x H x W
        # gamma
        self.gamma_value = np.full(shape_, self.gamma_init_val, self.dtype)
        gamma_gpu = gpuarray.to_gpu(self.gamma_value)
        self.gamma = TensorGPU(gamma_gpu, self.tensor_fmt, self.cudnn_dtype)
        # beta
        self.beta_value = np.full(shape_, self.beta_init_val, self.dtype)
        beta_gpu = gpuarray.to_gpu(self.beta_value)
        self.beta = TensorGPU(beta_gpu, self.tensor_fmt, self.cudnn_dtype)

        self.nparams = self.gamma.size + self.beta.size

        self.dgamma_cpu = np.zeros(self.gamma.ary.shape, self.dtype)
        self.dbeta_cpu = np.zeros(self.beta.ary.shape, self.dtype)

        if self.gpudirect:
            dgamma_gpu = drv.register_host_memory(self.dgamma_cpu)
            dbeta_gpu = drv.register_host_memory(self.dbeta_cpu)
        else:
            dgamma_gpu = gpuarray.empty(self.gamma.ary.shape, self.dtype)
            dbeta_gpu = gpuarray.empty(self.beta.ary.shape, self.dtype)

        self.dgamma = TensorGPU(dgamma_gpu, self.tensor_fmt, self.cudnn_dtype, gpudirect=self.gpudirect)
        self.dbeta = TensorGPU(dbeta_gpu, self.tensor_fmt, self.cudnn_dtype, gpudirect=self.gpudirect)

        running_mean_gpu = gpuarray.to_gpu(self.moving_mean_initializer(shape_, self.dtype))
        self.running_mean = TensorGPU(running_mean_gpu, self.tensor_fmt, self.cudnn_dtype)

        running_var_gpu = gpuarray.to_gpu(self.moving_variance_initializer(shape_, self.dtype))
        self.running_var = TensorGPU(running_var_gpu, self.tensor_fmt, self.cudnn_dtype)

        save_mean_gpu = gpuarray.empty(shape_, self.dtype)
        self.save_mean = TensorGPU(save_mean_gpu, self.tensor_fmt, self.cudnn_dtype)

        save_inv_var_gpu = gpuarray.empty(shape_, self.dtype)
        self.save_inv_var = TensorGPU(save_inv_var_gpu, self.tensor_fmt, self.cudnn_dtype)

        self.factor = 1.0 - self.momentum

    def forward(self, x):
        alpha, beta = 1.0, 0.0
        if self.model.mode == "train":
            cudnn.cudnnBatchNormalizationForwardTraining(self.cudnn_handle, self.mode, 
                alpha, beta, x.desc, x.ptr, 
                self.y.desc, self.y.ptr, self.gamma_beta_mean_var_desc, self.gamma.ptr, 
                self.beta.ptr, self.factor, self.running_mean.ptr, self.running_var.ptr, 
                self.epsilon, self.save_mean.ptr, self.save_inv_var.ptr)

        elif self.model.mode == "evaluate":
            cudnn.cudnnBatchNormalizationForwardInference(self.cudnn_handle, self.mode, 
                alpha, beta, x.desc, x.ptr, 
                self.y.desc, self.y.ptr, self.gamma_beta_mean_var_desc, self.gamma.ptr, 
                self.beta.ptr, self.running_mean.ptr, self.running_var.ptr, self.epsilon)
        return self.y

    def backward(self, prev_dx):
        alpha_dx, beta_dx, alpha_dgb, beta_dgb = 1.0, 0.0, 1.0, 0.0
        # Compute dx, dgamma, dbeta
        cudnn.cudnnBatchNormalizationBackward(self.cudnn_handle, self.mode, 
            alpha_dx, beta_dx, alpha_dgb, beta_dgb,
            self.x.desc, self.x.ptr, prev_dx.desc, prev_dx.ptr,
            self.dx.desc, self.dx.ptr, self.gamma_beta_mean_var_desc, 
            self.gamma.ptr, self.dgamma.ptr, self.dbeta.ptr, self.epsilon,
            self.save_mean.ptr, self.save_inv_var.ptr)

        if self.model.comm and not self.gpudirect:
            self.dgamma.ary.get_async(self.stream_2, self.dgamma_cpu)
            self.dbeta.ary.get_async(self.stream_2, self.dbeta_cpu)
        return self.dx


class AdditionBlockGPU(NN_layer.AdditionBlock):

    def initialize(self, prev_shape, need_dx, x):
        self.out_shapes = []
        need_dx = True
        self.x = x
        self.prev_shape = prev_shape
        for p_i, p in enumerate(self.paths):
            for i, l in enumerate(p):
                l.tracer = self.tracer
                l.dtype = self.dtype
                l.model = self.model
                l.batch_size = self.batch_size
                l.id = self.model.id + i
    
                l.cudnn_handle = self.cudnn_handle
                l.cublas_handle = self.cublas_handle
                l.gpudirect = self.gpudirect
                l.stream = self.stream
                l.cudnn_dtype = self.cudnn_dtype
                l.tensor_fmt = self.tensor_fmt
    
                l.initialize(prev_shape, need_dx, x)
                x = l.y
                if p_i == 0 and (len(p) - 1) == i: self.y = x
                prev_shape = l.shape

                self.fwd_time += l.fwd_time
                self.bwd_time += l.bwd_time
                self.nparams += l.nparams

            self.out_shapes.append(prev_shape)
            prev_shape = self.prev_shape            
            x = self.x
            self.model.id += len(p)
        
        self.model.id -= 1
        assert all([o == self.out_shapes[0] for o in self.out_shapes])
        self.shape = self.out_shapes[0]

    def forward(self, x):
        for i, p in enumerate(self.paths):
            y_i = x
            for l in p: y_i = l.forward(y_i)
            if i == 0: y = y_i
            else:
                alpha, beta = 1.0, 1.0
                cudnn.cudnnAddTensor(self.cudnn_handle, alpha, y_i.desc, 
                                     y_i.ptr, beta, y.desc, y.ptr)
        return y

    def backward(self, dy):
        for i, p in enumerate(self.paths):
            dx_i = dy
            for l in reversed(p): dx_i = l.backward(dx_i)
            if i == 0: dx = dx_i
            else:
                alpha, beta = 1.0, 1.0
                cudnn.cudnnAddTensor(self.cudnn_handle, alpha, dx_i.desc, 
                                     dx_i.ptr, beta, dx.desc, dx.ptr)
        return dx


class ConcatenationBlockGPU(NN_layer.ConcatenationBlock):

    def initialize(self, prev_shape, need_dx, x):
        need_dx = True
        self.out_shapes = []
        self.x = x
        self.prev_shape = prev_shape        
        for p in self.paths:
            for i, l in enumerate(p):
                l.tracer = self.tracer
                l.dtype = self.dtype
                l.model = self.model
                l.batch_size = self.batch_size
                l.id = self.model.id + i
    
                l.cudnn_handle = self.cudnn_handle
                l.cublas_handle = self.cublas_handle
                l.gpudirect = self.gpudirect
                l.stream = self.stream
                l.cudnn_dtype = self.cudnn_dtype
                l.tensor_fmt = self.tensor_fmt

                l.initialize(prev_shape, need_dx, x)
                x = l.y
                prev_shape = l.shape
                self.fwd_time += l.fwd_time
                self.bwd_time += l.bwd_time
                self.nparams += l.nparams

            self.out_shapes.append(prev_shape)
            prev_shape = self.prev_shape            
            x = self.x
            self.model.id += len(p)

        self.model.id -= 1
        assert all([o[1:] == self.out_shapes[0][1:] for o in self.out_shapes])
        self.out_co = [s[0] for s in self.out_shapes]
        self.idx_co = np.cumsum(self.out_co, axis=0)
        self.shape = (sum(self.out_co), *self.out_shapes[0][1:])

        self.concat = ElementwiseKernel(
            "T *dst, T *src, int N, int C, int H, int W, int first_c, int last_c".replace("T", 
                {"float32": "float", "float64": "double"}[self.dtype]),
            """int c_ = i / (H*W) % C;
               if (first_c <= c_ && c_ < last_c) {
                   int w_ = i % W;
                   int h_ = i / W % H;
                   int n_ = i / (C*H*W) % N;
                   int i_ = n_ * (last_c-first_c) * H * W + (c_-first_c) * H * W + h_ * W + w_;
                   dst[i] = src[i_];
               }
            """,
            "concat")

        self.split = ElementwiseKernel(
            "T *src, T *dst, int N, int C, int H, int W, int first_c, int last_c".replace("T", 
                {"float32": "float", "float64": "double"}[self.dtype]),
            """int c_ = i / (H*W) % C;
               if (first_c <= c_ && c_ < last_c) {
                   int w_ = i % W;
                   int h_ = i / W % H;
                   int n_ = i / (C*H*W) % N;
                   int i_ = n_ * (last_c-first_c) * H * W + (c_-first_c) * H * W + h_ * W + w_;
                   dst[i_] = src[i];
               }
            """,
            "split")

        # Activations y
        y_gpu = gpuarray.empty((self.batch_size, *self.shape), self.dtype)
        self.y = TensorGPU(y_gpu, self.tensor_fmt, self.cudnn_dtype)

        # Derivative dx
        self.dy = []
        for i, p in enumerate(self.paths):
            dy_gpu = gpuarray.empty((self.batch_size, *self.out_shapes[i]), self.dtype)
            self.dy.append(TensorGPU(dy_gpu, self.tensor_fmt, self.cudnn_dtype))

    def forward(self, x):
        for i, p in enumerate(self.paths):
            x_i = x
            for l in p: x_i = l.forward(x_i)
            self.concat(self.y.ary, x_i.ary, self.batch_size, *self.shape, 
                        0 if i == 0 else self.idx_co[i-1], self.idx_co[i])
        return self.y

    def backward(self, dy):
        for i, p in enumerate(self.paths):
            self.split(dy.ary, self.dy[i].ary, self.batch_size, *self.shape, 
                        0 if i == 0 else self.idx_co[i-1], self.idx_co[i])
            dx_i = self.dy[i]
            for l in reversed(p): dx_i = l.backward(dx_i)
            if i == 0: dx = dx_i
            else:
                alpha, beta = 1.0, 1.0
                cudnn.cudnnAddTensor(self.cudnn_handle, alpha, dx_i.desc, 
                                     dx_i.ptr, beta, dx.desc, dx.ptr)
        return dx
