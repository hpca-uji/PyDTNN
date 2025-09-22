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
#  with this program.  If not, see <https://www.gnu.org/licenses/>.
#

import numpy as np
# noinspection PyUnresolvedReferences
import pycuda.gpuarray as gpuarray
# noinspection PyUnresolvedReferences
from pycuda.compiler import SourceModule
# noinspection PyUnresolvedReferences
from pycuda.elementwise import ElementwiseKernel

from pydtnn.backends.gpu.optimizers.optimizer_gpu import OptimizerGPU, gpuarray_t
from pydtnn.optimizers import RMSProp
from pydtnn.backends.gpu.layers import LayerGPU
from pydtnn.backends.gpu import TensorGPU

class RMSPropGPU(OptimizerGPU, RMSProp):
    """
    RMSPropGPU Optimizer
    """

    def __init__(self, learning_rate=1e-2, rho=0.9, epsilon=1e-7, decay=0.0, dtype=np.float32):
        super().__init__(learning_rate, rho, epsilon, decay, dtype)

        self.update_gpu = ElementwiseKernel("T *w, T *dw, T *cache, \
                               float lr, float decay, float rho, float epsilon".replace("T",
                                                                                        {np.float32: "float",
                                                                                         np.float64: "double"}[dtype]),
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


    def initialize(self, list_layers: list[LayerGPU]) -> None:
        for layer in list_layers:
            list_grad_vars = list(layer.grad_vars.keys())
                    
            if len(list_grad_vars) != 0:
                self.context[layer] = dict[str, gpuarray_t]()
                for w_ in list_grad_vars:
                    w = getattr(layer, w_)
                    self.context[layer]["cache_%s" % w_] = gpuarray.zeros_like(w.ary, dtype=layer.model.dtype)



    def update(self, layer: LayerGPU):
        for w_, dw_ in layer.grad_vars.items():
            w, dw = getattr(layer, w_), getattr(layer, dw_)
            cache = self.context[layer]["cache_%s" % w_]
            w:TensorGPU
            dw:TensorGPU
            cache:gpuarray

            if self.gpudirect:
                n = self.get_batch_size(w)
                threads, blocks = self.get_threads_and_blocks()

                self.update_gpudirect(w.ary.gpudata, dw.ptr_intp, cache.gpudata,
                                      np.float32(self.learning_rate),
                                      np.float32(self.decay), np.float32(self.rho),
                                      np.float32(self.epsilon), np.int32(n),
                                      grid=(int(blocks), 1, 1), block=(int(threads), 1, 1),
                                      stream=layer.stream_2)
            else:
                self.update_gpu(w.ary, dw.ary, cache, np.float32(self.learning_rate),
                                np.float32(self.decay), np.float32(self.rho),
                                np.float32(self.epsilon), stream=layer.stream_2)
