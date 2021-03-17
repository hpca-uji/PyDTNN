#
#  This file is part of Python Distributed Training of Neural Networks (PyDTNN)
#
#  Copyright (C) 2021 Universitat Jaume I
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
import pycuda.gpuarray as gpuarray
from pycuda.compiler import SourceModule
from pycuda.elementwise import ElementwiseKernel

from pydtnn import optimizers


class NadamGPU(optimizers.Nadam):
    """
    NadamGPU optimizer
    """

    def __init__(self, learning_rate=1e-2, beta1=0.99, beta2=0.999, epsilon=1e-7, decay=0.0, dtype=np.float32):
        super().__init__(learning_rate, beta1, beta2, epsilon, decay, dtype)

        self.update_gpu = ElementwiseKernel("T *w, T *dw, T *m, T *v, \
                               float it, float lr, float decay, \
                               float beta1, float beta2, float epsilon".replace("T",
                                                                                {np.float32: "float",
                                                                                 np.float64: "double"}[dtype]),
                                            "m[i] = beta1 * m[i] + (1 - beta1) * dw[i]; \
                                             v[i] = beta2 * v[i] + (1 - beta2) * pow(dw[i], 2); \
                                             w[i] -= lr * (decay * w[i] + (((m[i] + (1 - beta1) * dw[i]) / \
                                                     (1 - pow(beta1, it))) / \
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

        for w_, dw_ in layer.grad_vars.items():
            w, dw = getattr(layer, w_), getattr(layer, dw_)
            m = getattr(layer, "m_%s" % w_, gpuarray.zeros_like(w.ary, dtype=layer.model.dtype))
            v = getattr(layer, "v_%s" % w_, gpuarray.zeros_like(w.ary, dtype=layer.model.dtype))

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
                                      grid=(int(blocks), 1, 1), block=(int(threads), 1, 1),
                                      stream=layer.stream_2)
            else:
                self.update_gpu(w.ary, dw.ary, m, v,
                                np.float32(it), np.float32(self.learning_rate),
                                np.float32(self.decay), np.float32(self.beta1),
                                np.float32(self.beta2), np.float32(self.epsilon),
                                stream=layer.stream_2)

            if not hasattr(layer, "m_%s" % w_) and not hasattr(layer, "v_%s" % w_):
                setattr(layer, "m_%s" % w_, m)
                setattr(layer, "v_%s" % w_, v)
