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
__maintainer__ = "Manuel F. Dolz"
__status__ = "Production"
__version__ = "1.1.0"


import numpy as np
import activation
from layer import Layer
from NN_relu_cython import relu_cython
from util import TensorGPU

try:
    #import pycuda.autoinit
    import pycuda.gpuarray as gpuarray
    import pycuda.driver as drv
    import libcudnn.libcudnn as cudnn
    import ctypes
    from skcuda import cublas
except:
    pass

class SigmoidGPU(activation.Sigmoid):

    def initialize(self, prev_shape, need_dx, x):
        self.shape = prev_shape
        self.need_dx = need_dx
        self.x = x

        self.act_desc = cudnn.cudnnCreateActivationDescriptor()

        mode = cudnn.cudnnActivationMode['CUDNN_ACTIVATION_SIGMOID']
        nan  = cudnn.cudnnNanPropagation['CUDNN_NOT_PROPAGATE_NAN']
        cudnn.cudnnSetActivationDescriptor(self.act_desc, mode, nan, 0.0)
            
        # Activations a
        y_gpu = gpuarray.empty(x.ary.shape, self.dtype)
        self.y = TensorGPU(y_gpu, self.tensor_fmt, self.cudnn_dtype)
        
        if self.need_dx:
            # Derivative dx
            dx_gpu = gpuarray.empty(x.ary.shape, self.dtype)
            self.dx = TensorGPU(dx_gpu, self.tensor_fmt, self.cudnn_dtype)

    def forward(self, x):
        alpha, beta = 1.0, 0.0
        cudnn.cudnnActivationForward(self.cudnn_handle, self.act_desc, alpha, 
                                     x.desc, x.ptr, beta, 
                                     self.y.desc, self.y.ptr)
        return self.y

    def backward(self, dy):
        if self.need_dx:
            alpha, beta = 1.0, 0.0
            cudnn.cudnnActivationBackward(self.cudnn_handle, self.act_desc, alpha, 
                                     self.y.desc, self.y.ptr, 
                                     dy.desc, dy.ptr, 
                                     self.x.desc, self.x.ptr, beta, 
                                     self.dx.desc, self.dx.ptr)
            return self.dx


class ReluGPU(activation.Relu):

    def initialize(self, prev_shape, need_dx, x): 
        self.shape = prev_shape
        self.need_dx = need_dx
        self.x = x

        self.act_desc = cudnn.cudnnCreateActivationDescriptor()

        mode = cudnn.cudnnActivationMode['CUDNN_ACTIVATION_RELU']
        nan  = cudnn.cudnnNanPropagation['CUDNN_NOT_PROPAGATE_NAN']

        # We set the maximum value to the relu to 0, which specifies the upper bound
        relu_ceiling = 0.0
        cudnn.cudnnSetActivationDescriptor(self.act_desc, mode, nan, relu_ceiling)

        # Activations a
        y_gpu = gpuarray.empty(x.ary.shape, self.dtype)
        self.y = TensorGPU(y_gpu, self.tensor_fmt, self.cudnn_dtype)
     
        if self.need_dx:
            # Derivative dx
            dx_gpu = gpuarray.empty(x.ary.shape, self.dtype)
            self.dx = TensorGPU(dx_gpu, self.tensor_fmt, self.cudnn_dtype)

    def forward(self, x):
        alpha, beta = 1.0, 0.0
        cudnn.cudnnActivationForward(self.cudnn_handle, self.act_desc, alpha, 
                                     x.desc, x.ptr, beta, 
                                     self.y.desc, self.y.ptr)
        return self.y

    def backward(self, dy):
        if self.need_dx:
            alpha, beta = 1.0, 0.0
            cudnn.cudnnActivationBackward(self.cudnn_handle, self.act_desc, alpha, 
                                     self.y.desc, self.y.ptr, 
                                     dy.desc, dy.ptr, 
                                     self.x.desc, self.x.ptr, beta,
                                     self.dx.desc, self.dx.ptr)
            return self.dx


class TanhGPU(activation.Tanh):

    def initialize(self, prev_shape, need_dx, x):
        self.shape = prev_shape
        self.need_dx = need_dx
        self.x = x

        self.act_desc = cudnn.cudnnCreateActivationDescriptor()

        mode = cudnn.cudnnActivationMode['CUDNN_ACTIVATION_TANH']
        nan  = cudnn.cudnnNanPropagation['CUDNN_NOT_PROPAGATE_NAN']
        cudnn.cudnnSetActivationDescriptor(self.act_desc, mode, nan, 0.0)            

        # Activations a
        y_gpu = gpuarray.empty(x.ary.shape, self.dtype)
        self.y = TensorGPU(y_gpu, self.tensor_fmt, self.cudnn_dtype)
        
        if self.need_dx:
            # Derivative dx
            dx_gpu = gpuarray.empty(x.ary.shape, self.dtype)
            self.dx = TensorGPU(dx_gpu, self.tensor_fmt, self.cudnn_dtype)

    def forward(self, x):
        alpha, beta = 1.0, 0.0
        cudnn.cudnnActivationForward(self.cudnn_handle, self.act_desc, alpha, 
                                     x.desc, x.ptr, beta, 
                                     self.y.desc, self.y.ptr)
        return self.y

    def backward(self, dy):
        if self.need_dx:
            alpha, beta = 1.0, 0.0
            cudnn.cudnnActivationBackward(self.cudnn_handle, self.act_desc, alpha, 
                                     self.y.desc, self.y.ptr, 
                                     dy.desc, dy.ptr, 
                                     self.x.desc, self.x.ptr, beta, 
                                     self.dx.desc, self.dx.ptr)
            return self.dx


class ArctanhGPU(activation.Arctanh):

    def initialize(self, prev_shape, need_dx, x):
        self.shape = prev_shape
        self.need_dx = need_dx

        self.atanh  = ElementwiseKernel("T *in, T *out".replace("T",
                            {np.float32: "float", np.float64: "double"}[self.dtype]),
                            "out[i] = %s(in[i]);" % \
                            {np.float32: "atanhf", np.float64: "atanh"}[self.dtype], "atanh")

        self.datanh = ElementwiseKernel("T *in, T *out".replace("T",
                            {np.float32: "float", np.float64: "double"}[self.dtype]),
                            "out[i] = 1.0 / (1.0 + %s(in[i], 2));" % \
                            {np.float32: "powf", np.float64: "pow"}[self.dtype], "datanh")

        # Activations a
        y_gpu = gpuarray.empty(x.ary.shape, self.dtype)
        self.y = TensorGPU(y_gpu, self.tensor_fmt, self.cudnn_dtype)
       
        if self.need_dx:  
            # Derivative dx
            dx_gpu = gpuarray.empty(x.ary.shape, self.dtype)
            self.dx = TensorGPU(dx_gpu, self.tensor_fmt, self.cudnn_dtype)

    def forward(self, x):
        self.atanh(x.ary, self.y.ary, stream=self.stream)
        return self.y

    def backward(self, dy):
        if self.need_dx:
            # Compute dx
            self.datanh(dy.ary, self.dx.ary, stream=self.stream)
            return self.dx 


class LogGPU(activation.Log):

    def initialize(self, prev_shape, need_dx, x):
        self.shape = prev_shape
        self.need_dx = need_dx

        self.log  = ElementwiseKernel("T *in, T *out".replace("T",
                            {np.float32: "float", np.float64: "double"}[self.dtype]),
                            "out[i] = %s(1.0 / (1.0 + %s(-in[i])));" % \
                            ({np.float32: "logf", np.float64: "log"}[self.dtype],
                             {np.float32: "expf", np.float64: "exp"}[self.dtype]), "log")

        self.dlog = ElementwiseKernel("T *in, T *out".replace("T",
                            {np.float32: "float", np.float64: "double"}[self.dtype]),
                            "out[i] = 1.0 / (1.0 + %s(in[i]));" % \
                            {np.float32: "expf", np.float64: "exp"}[self.dtype], "log")

        # Activations a
        y_gpu = gpuarray.empty(x.ary.shape, self.dtype)
        self.y = TensorGPU(y_gpu, self.tensor_fmt, self.cudnn_dtype)

        if self.need_dx:
            # Derivative dx
            dx_gpu = gpuarray.empty(x.ary.shape, self.dtype)
            self.dx = TensorGPU(dx_gpu, self.tensor_fmt, self.cudnn_dtype)

    def forward(self, x):
        self.log(x.ary, self.y.ary, stream=self.stream)
        return self.y

    def backward(self, dy):
        if self.need_dx:
            # Compute dx
            self.dlog(dy.ary, self.dx.ary, stream=self.stream)
            return self.dx

    
class SoftmaxGPU(activation.Softmax):

    def initialize(self, prev_shape, need_dx, x):
        self.shape = prev_shape
        self.need_dx = need_dx

        self.mode = cudnn.cudnnSoftmaxMode['CUDNN_SOFTMAX_MODE_INSTANCE']
        self.algo = cudnn.cudnnSoftmaxAlgorithm['CUDNN_SOFTMAX_ACCURATE']

        # Activations a
        y_gpu = gpuarray.empty(x.ary.shape, self.dtype)
        self.y = TensorGPU(y_gpu, self.tensor_fmt, self.cudnn_dtype)
        
        # Derivative dx
        dx_gpu = gpuarray.empty(x.ary.shape, self.dtype)
        self.dx = TensorGPU(dx_gpu, self.tensor_fmt, self.cudnn_dtype)

    def forward(self, x):
        alpha, beta = 1.0, 0.0
        cudnn.cudnnSoftmaxForward(self.cudnn_handle, self.algo, self.mode, alpha, 
                                  x.desc, x.ptr, beta, 
                                  self.y.desc, self.y.ptr)
        return self.y

    def backward(self, dy):
        if self.need_dx:
            alpha, beta = 1.0, 0.0
            cudnn.cudnnSoftmaxBackward(self.cudnn_handle, self.algo, self.mode, alpha,
                                  self.y.desc, self.y.ptr, 
                                  dy.desc, dy.ptr, beta, 
                                  self.dx.desc, self.dx.ptr)
            return self.dx
