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
#  with this program.  If not, see <https://www.gnu.org/licenses/>.
#

import numpy as np
# noinspection PyUnresolvedReferences
import pycuda.gpuarray as gpuarray
# noinspection PyUnresolvedReferences
from pycuda.compiler import SourceModule
# noinspection PyUnresolvedReferences
from pycuda.elementwise import ElementwiseKernel

from pydtnn.backends.gpu.optimizers.optimizer_gpu import OptimizerGPU
from pydtnn.optimizers import SGD


class SGDGPU(OptimizerGPU, SGD):
    """
    SGDGPU optimizer
    """

    def __init__(self, learning_rate=1e-2, momentum=0.9, nesterov=False, decay=0.0, dtype=np.float32):
        super().__init__(learning_rate, momentum, nesterov, decay, dtype)

        self.update_gpu = ElementwiseKernel("T *w, T * dw, T * v, \
                               float lr, float decay, float momentum".replace("T",
                                                                              {np.float32: "float",
                                                                               np.float64: "double"}[dtype]),
                                            "v[i] = momentum * v[i] + dw[i]; %s;" %
                                            ({True: "w[i] -= lr * (decay * w[i] + dw[i] + momentum * v[i])",
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
             }""" % ({True: "w[i] -= lr * (decay * w[i] + dw[i] + momentum * v[i])",
                      False: "w[i] -= lr * (decay * w[i] + v[i])"}[nesterov])).replace("T",
                                                                                       {np.float32: "float",
                                                                                        np.float64: "double"}[dtype])
                                             ).get_function("SGD_kernel")

    def update(self, layer, **kwargs):
        for w_, dw_ in layer.grad_vars.items():
            w, dw = getattr(layer, w_), getattr(layer, dw_)
            velocity = getattr(layer, "velocity_%s" % w_, gpuarray.zeros_like(w.ary, dtype=layer.model.dtype))

            if self.gpudirect:
                rows, cols = w.shape[0], np.prod(w.shape[1:])

                # threads, blocks = 128, 10240
                # assert threads * blocks >= rows * cols
                threads = 1024
                blocks = (rows * cols) // threads + 1

                self.update_gpudirect(w.ary.gpudata, dw.ptr_intp, velocity.gpudata,
                                      np.float32(self.learning_rate), np.float32(self.decay),
                                      np.float32(self.momentum), np.int32(rows * cols),
                                      grid=(int(blocks), 1, 1), block=(int(threads), 1, 1),
                                      stream=layer.stream_2)

            else:
                self.update_gpu(w.ary, dw.ary, velocity, np.float32(self.learning_rate),
                                np.float32(self.decay), np.float32(self.momentum),
                                stream=layer.stream_2)

            if not hasattr(layer, "velocity_%s" % w_):
                setattr(layer, "velocity_%s" % w_, velocity)
