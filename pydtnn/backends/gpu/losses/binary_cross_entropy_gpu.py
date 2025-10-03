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

import numpy as np
# noinspection PyUnresolvedReferences
import pycuda.gpuarray as gpuarray
# noinspection PyUnresolvedReferences
from pycuda.compiler import SourceModule
# noinspection PyUnresolvedReferences
from pycuda.driver import Function

from pydtnn.losses import BinaryCrossEntropy
from .loss_gpu import LossGPU
from pydtnn.backends.gpu import TensorGPU

class BinaryCrossEntropyGPU(LossGPU, BinaryCrossEntropy):

    def __init_gpu_kernel__(self) -> Function:
        module = SourceModule("""
        __global__ void binary_cross_entropy(T *y_targ, T *y_pred, T *res,
                                             T *dx, int b, int n, T eps)
        {
            int idx = blockIdx.x * blockDim.x + threadIdx.x;
            if (idx < b) {
                int i = 0, max = 0;
                T pred;
                res[idx] = 0;
                for ( i = 0; i < n; i++ ) {
                    res[idx]+= logf(fmaxf((1 - y_targ[idx * n + i] ) -
                                               y_pred[idx * n + i], eps));
                    pred = y_pred[idx * n + max];
                    if ( pred < eps )          pred = eps;
                    else if ( pred > (1-eps) ) pred = (1-eps);
                    dx[idx * n + i] = (-(y_targ[idx * n + i]  / pred) +
                                   ((1 - y_targ[idx * n + i]) / pred) ) / b;
                }
            }
            return;
        }
        """.replace("T", {np.float32: "float", np.float64: "double"}[self.model.dtype]))
        return module.get_function("binary_cross_entropy")

    def __call__(self, y_pred: TensorGPU, y_targ:TensorGPU, batch_size:int) -> tuple[float, TensorGPU]:
        assert len(y_targ.shape) == 2
        threads, blocks = self.get_threads_and_blocks()
        self.kernel(y_targ, y_pred, self.loss, self.dx.ary,
                    batch_size, self.shape[1], self.eps,
                    grid=(blocks, 1, 1), block=(threads, 1, 1),
                    stream=self.model.stream)
        loss:float = -gpuarray.sum(self.loss[:batch_size]) / batch_size
        return loss, self.dx
