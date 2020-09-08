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

import ctypes
import numpy as np
import NN_optimizer
from NN_layer import Layer

import pycuda.gpuarray as gpuarray
import pycuda.driver as drv
from pycuda.elementwise import ElementwiseKernel
from pycuda.compiler import SourceModule


class SGDGPU(NN_optimizer.SGD):

    def __init__(self, learning_rate=1e-2, momentum=0.9, 
                 nesterov=False, decay=0.0, dtype=np.float32):
        super(SGDGPU, self).__init__(learning_rate, momentum, nesterov, decay, dtype)

        self.update_gpu = ElementwiseKernel("T *w, T * dw, T * v, \
                               float lr, float decay, float momentum".replace("T", 
                                   {np.float32: "float", np.float64: "double"}[dtype]), 
                               "v[i] = momentum * v[i] + dw[i]; %s;" % 
                                  ({True:  "w[i] -= lr * (decay * w[i] + dw[i] + momentum * v[i])",
                                    False: "w[i] -= lr * (decay * w[i] + v[i])"}[nesterov]),
                               "SGD_kernel")

        self.update_gpudirect = SourceModule(("""
            __global__ void SGD_kernel(T *w, T *dw, T *v, 
                               float lr, float decay, float momentum, int N) {
                int i = blockIdx.x * blockDim.x + threadIdx.x;
                if (i < N) {
                    v[i] = momentum * v[i] + dw[i];
                    %s;
                }
             }""" % ({True:  "w[i] -= lr * (decay * w[i] + dw[i] + momentum * v[i])",
                      False: "w[i] -= lr * (decay * w[i] + v[i])"}[nesterov])).replace("T", 
                     {np.float32: "float", np.float64: "double"}[dtype])
             ).get_function("SGD_kernel")

    def update(self, layer):
        layer.stream_2.synchronize()      

        for w_, dw_ in layer.grad_vars.items():
            w, dw = getattr(layer, w_), getattr(layer, dw_)
            velocity = getattr(layer, "velocity_%s" % (w_), gpuarray.zeros_like(w.ary, dtype=layer.dtype))

            if self.gpudirect:
                rows, cols = w.shape[0], np.prod(w.shape[1:])

                # threads, blocks = 128, 10240
                # assert threads * blocks >= rows * cols
                threads = 1024
                blocks = (rows * cols) // threads + 1
                
                self.update_gpudirect(w.ary.gpudata, dw.ptr_intp, velocity.gpudata, 
                                      np.float32(self.learning_rate), np.float32(self.decay), 
                                      np.float32(self.momentum), np.int32(rows * cols),
                                      grid=(int(blocks),1,1), block=(int(threads),1,1), 
                                      stream=layer.stream)

            else:
                self.update_gpu(w.ary, dw.ary, velocity, np.float32(self.learning_rate), 
                                np.float32(self.decay), np.float32(self.momentum), 
                                stream=layer.stream)

            if not hasattr(layer, "velocity_%s" % (w_)):
                setattr(layer, "velocity_%s" % (w_), velocity)


class RMSPropGPU(NN_optimizer.RMSProp):

    def __init__(self, learning_rate=1e-2, rho=0.9, 
                 epsilon=1e-7, decay=0.0, dtype=np.float32):
        super(RMSPropGPU, self).__init__(learning_rate, rho, epsilon, decay, dtype)

        self.update_gpu = ElementwiseKernel("T *w, T *dw, T *cache, \
                               float lr, float decay, float rho, float epsilon".replace("T",
                                   {np.float32: "float", np.float64: "double"}[dtype]),
                               "cache[i] = rho * cache[i] + (1 - rho) * pow(dw[i], 2); \
                                w[i] -= lr * (decay * w[i] + (dw[i] / sqrtf(cache[i] + epsilon)))".
                                   replace("pow", {np.float32: "powf", np.float64: "pow"}[dtype]),
                               "RMSProp_kernel")

        self.update_gpudirect = SourceModule("""
            __global__ void RMSProp_kernel(T *w, T *dw, T *cache,
                                float lr, float decay, float rho, float epsilon, int N) {
                int i = blockIdx.x * blockDim.x + threadIdx.x;
                if (i < N) {
                    cache[i] = rho * cache[i] + (1 - rho) * pow(dw[i], 2);
                    w[i] -= lr * (decay * w[i] + (dw[i] / sqrt(cache[i] + epsilon)));
                }
            }""".replace("T", {np.float32: "float", np.float64: "double"}[dtype]).
                 replace("pow", {np.float32: "powf", np.float64: "pow"}[dtype])
        ).get_function("RMSProp_kernel")

    def update(self, layer):
        layer.stream_2.synchronize()

        for w_, dw_ in layer.grad_vars.items():
            w, dw = getattr(layer, w_), getattr(layer, dw_)
            cache = getattr(layer, "cache_%s" % (w_), gpuarray.zeros_like(w.ary, dtype=layer.dtype))

            if self.gpudirect:
                rows, cols = w.shape[0], np.prod(w.shape[1:])

                # threads, blocks = 128, 10240
                # assert threads * blocks >= rows * cols
                threads = 1024
                blocks = (rows * cols) // threads + 1
                
                self.update_gpudirect(w.ary.gpudata, dw.ptr_intp, cache.gpudata, 
                                      np.float32(self.learning_rate),
                                      np.float32(self.decay), np.float32(self.rho), 
                                      np.float32(self.epsilon), np.int32(rows * cols),
                                      grid=(int(blocks),1,1), block=(int(threads),1,1), 
                                      stream=layer.stream)
            else:
                self.update_gpu(w.ary, dw.ary, cache, np.float32(self.learning_rate), 
                                np.float32(self.decay), np.float32(self.rho), 
                                np.float32(self.epsilon), stream=layer.stream)

            if not hasattr(layer, "cache_%s" % (w_)):
                setattr(layer, "cache_%s" % (w_), cache)


class AdamGPU(NN_optimizer.Adam):

    def __init__(self, learning_rate=1e-2, beta1=0.99, beta2=0.999, 
                 epsilon=1e-7, decay=0.0, dtype=np.float32):
        super(AdamGPU, self).__init__(learning_rate, beta1, beta2, epsilon, decay, dtype)

        self.update_gpu = ElementwiseKernel("T *w, T *dw, T *m, T *v, \
                               float it, float lr, float decay, \
                               float beta1, float beta2, float epsilon".replace("T", 
                                   {np.float32: "float", np.float64: "double"}[dtype]),
                               "m[i] = beta1 * m[i] + (1 - beta1) * dw[i]; \
                                v[i] = beta2 * v[i] + (1 - beta2) * pow(dw[i], 2); \
                                w[i] -= lr * (decay * w[i] + ((m[i] / (1 - pow(beta1, it))) / \
                                                          sqrt(v[i] / (1 - pow(beta2, it)) + epsilon)))".
                                   replace("pow", {np.float32: "powf", np.float64: "pow"}[dtype]),
                               "Adam_kernel")

        self.update_gpudirect = SourceModule("""
            __global__ void Adam_kernel(T *w, T *dw, T *m, T *v,
                               float it, float lr, float decay,
                               float beta1, float beta2, float epsilon, int N) {
                int i = blockIdx.x * blockDim.x + threadIdx.x;
                if (i < N) {
                    m[i] = beta1 * m[i] + (1 - beta1) * dw[i];
                    v[i] = beta2 * v[i] + (1 - beta2) * pow(dw[i], 2);
                    w[i] -= lr * (decay * w[i] + ((m[i] / (1 - pow(beta1, it))) / 
                                              sqrt(v[i] / (1 - pow(beta2, it)) + epsilon)));
                }
            }""".replace("T", {np.float32: "float", np.float64: "double"}[dtype]).
                 replace("pow", {np.float32: "powf", np.float64: "pow"}[dtype]),
        ).get_function("Adam_kernel")

    def update(self, layer):
        it = getattr(layer, "it", 0) + 1
        setattr(layer, "it", it)
        layer.stream_2.synchronize()

        for w_, dw_ in layer.grad_vars.items():
            w, dw = getattr(layer, w_), getattr(layer, dw_)
            m = getattr(layer, "m_%s" % (w_), gpuarray.zeros_like(w.ary, dtype=layer.dtype))
            v = getattr(layer, "v_%s" % (w_), gpuarray.zeros_like(w.ary, dtype=layer.dtype))

            if self.gpudirect:
                rows, cols = w.shape[0], np.prod(w.shape[1:])

                # threads, blocks = 128, 10240
                # assert threads * blocks >= rows * cols
                threads = 1024
                blocks = (rows * cols) // threads + 1

                self.update_gpudirect(w.ary.gpudata, dw.ptr_intp, m.gpudata, v.gpudata,
                                      np.float32(it), np.float32(self.learning_rate), 
                                      np.float32(self.decay), np.float32(self.beta1), 
                                      np.float32(self.beta2), np.float32(self.epsilon),
                                      np.int32(rows * cols),
                                      grid=(int(blocks),1,1), block=(int(threads),1,1), 
                                      stream=layer.stream)
            else:
                self.update_gpu(w.ary, dw.ary, m, v, 
                                np.float32(it), np.float32(self.learning_rate), 
                                np.float32(self.decay), np.float32(self.beta1), 
                                np.float32(self.beta2), np.float32(self.epsilon),
                                stream=layer.stream)

            if not hasattr(layer, "m_%s" % (w_)) and not hasattr(layer, "v_%s" % (w_)):
                setattr(layer, "m_%s" % (w_), m)
                setattr(layer, "v_%s" % (w_), v)


class NadamGPU(NN_optimizer.Nadam):

    def __init__(self, learning_rate=1e-2, beta1=0.99, beta2=0.999, 
                 epsilon=1e-7, decay=0.0, dtype=np.float32):
        super(NadamGPU, self).__init__(learning_rate, beta1, beta2, epsilon, decay, dtype)


        self.update_gpu = ElementwiseKernel("T *w, T *dw, T *m, T *v, \
                               float it, float lr, float decay, \
                               float beta1, float beta2, float epsilon".replace("T", 
                                   {np.float32: "float", np.float64: "double"}[dtype]),
                               "m[i] = beta1 * m[i] + (1 - beta1) * dw[i]; \
                                v[i] = beta2 * v[i] + (1 - beta2) * pow(dw[i], 2); \
                                w[i] -= lr * (decay * w[i] + (((m[i] + (1 - beta1) * dw[i]) / (1 - pow(beta1, it))) / \
                                        sqrtf((v[i] / (1 - pow(beta2, it))) + epsilon)))".
                                   replace("pow", {np.float32: "powf", np.float64: "pow"}[dtype]),
                               "Nadam_kernel")

        self.update_gpudirect = SourceModule("""
            __global__ void Nadam_kernel(T *w, T *dw, T *m, T *v,
                               float it, float lr, float decay, 
                               float beta1, float beta2, float epsilon, int N) {
                int i = blockIdx.x * blockDim.x + threadIdx.x;
                if (i < N) {
                    m[i] = beta1 * m[i] + (1 - beta1) * dw[i];
                    v[i] = beta2 * v[i] + (1 - beta2) * pow(dw[i], 2);
                    w[i] -= lr * (decay * w[i] + (((m[i] + (1 - beta1) * dw[i]) / (1 - pow(beta1, it))) /
                                               sqrt(v[i] / (1 - pow(beta2, it)) + epsilon)));
                }
            }""".replace("T", {np.float32: "float", np.float64: "double"}[dtype]).
                 replace("pow", {np.float32: "powf", np.float64: "pow"}[dtype]),
        ).get_function("Nadam_kernel")

    def update(self, layer):
        it = getattr(layer, "it", 0) + 1
        setattr(layer, "it", it)
        layer.stream_2.synchronize()

        for w_, dw_ in layer.grad_vars.items():
            w, dw = getattr(layer, w_), getattr(layer, dw_)
            m = getattr(layer, "m_%s" % (w_), gpuarray.zeros_like(w.ary, dtype=layer.dtype))
            v = getattr(layer, "v_%s" % (w_), gpuarray.zeros_like(w.ary, dtype=layer.dtype))

            if self.gpudirect:
                rows, cols = w.shape[0], np.prod(w.shape[1:])

                # threads, blocks = 128, 10240
                # assert threads * blocks >= rows * cols
                threads = 1024
                blocks = (rows * cols) // threads + 1

                self.update_gpudirect(w.ary.gpudata, dw.ptr_intp, m.gpudata, v.gpudata,
                                      np.float32(it), np.float32(self.learning_rate),
                                      np.float32(self.decay), np.float32(self.beta1),
                                      np.float32(self.beta2), np.float32(self.epsilon),
                                      np.int32(rows * cols),
                                      grid=(int(blocks),1,1), block=(int(threads),1,1),
                                      stream=layer.stream)
            else:
                update_gpu(w.ary, dw.ary, m, v, 
                           np.float32(it), np.float32(self.learning_rate), 
                           np.float32(self.decay), np.float32(self.beta1), 
                           np.float32(self.beta2), np.float32(self.epsilon),
                           stream=layer.stream)

            if not hasattr(layer, "m_%s" % (w_)) and not hasattr(layer, "v_%s" % (w_)):
                setattr(layer, "m_%s" % (w_), m)
                setattr(layer, "v_%s" % (w_), v)


# Compatibility aliases

sgd_gpu = SGDGPU
rmsprop_gpu = RMSPropGPU
adam_gpu = AdamGPU
nadam_gpu = NadamGPU
