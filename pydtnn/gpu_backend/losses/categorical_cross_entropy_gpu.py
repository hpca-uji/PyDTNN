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
#  with this program. If not, see <https://www.gnu.org/licenses/>.
#

import numpy as np
# noinspection PyUnresolvedReferences
import pycuda.gpuarray as gpuarray
# noinspection PyUnresolvedReferences
from pycuda.compiler import SourceModule

from .loss_gpu import LossGPU


class CategoricalCrossEntropyGPU(LossGPU):

    def __init_gpu_kernel__(self):
        module = SourceModule("""
        __global__ void categorical_cross_entropy(T *y_targ, T *y_pred, T *res,
                                                  T *dx, int b, int bs, int n, float eps)
        {
            int idx = blockIdx.x * blockDim.x + threadIdx.x;
            if (idx < b) {
                int i = 0, max = 0;
                T max_value = y_targ[idx * n];
                dx[idx * n] = y_targ[idx * n];
                for ( i = 1; i < n; i++ ) {
                    dx[idx * n + i] = y_targ[idx * n + i];
                    if ( y_targ[idx * n + i] > max_value ) {
                        max = i;
                        max_value = y_targ[idx * n + i];
                    }
                }
                T pred = y_pred[idx * n + max];
                if ( pred < eps )          pred = eps;
                else if ( pred > (1-eps) ) pred = (1-eps);
                res[idx] = logf(pred);
                dx[idx * n + max] /= -(pred * bs);
            }
            return;
        }
        """.replace("T", {np.float32: "float", np.float64: "double"}[self.dtype]))
        return module.get_function("categorical_cross_entropy")

    def __call__(self, y_pred, y_targ, global_batch_size):
        threads = min(self.b, 1024)
        blocks = max(self.b, 1024) // threads + 1
        self.kernel(y_targ.ary, y_pred.ary, self.loss, self.dx.ary,
                    np.int32(self.b), np.int32(global_batch_size),
                    np.int32(self.n), np.float32(self.eps),
                    grid=(blocks, 1, 1), block=(threads, 1, 1),
                    stream=self.model.stream)
        loss = -gpuarray.sum(self.loss).get() / self.b
        return loss, self.dx
