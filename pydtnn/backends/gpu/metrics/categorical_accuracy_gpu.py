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

from pydtnn.metrics import CategoricalAccuracy
from .metric_gpu import MetricGPU


class CategoricalAccuracyGPU(MetricGPU, CategoricalAccuracy):

    def __init_gpu_kernel__(self):
        module = SourceModule("""
        __global__ void categorical_accuracy(T *y_targ, T *y_pred, T *res, int b, int n)
        {
            int idx = blockIdx.x * blockDim.x + threadIdx.x;
            if (idx < b){
                int i = 0, max = 0;
                T max_value = y_pred[idx * n];
                for ( i = 1; i < n; i++ ) {
                    if ( y_pred[idx * n + i] > max_value ){
                        max = i;
                        max_value = y_pred[idx * n + i];
                    }
                }
                res[idx] = y_targ[idx * n + max];
            }
            return;
        }
        """.replace("T", {np.float32: "float", np.float64: "double"}[self.model.dtype]))
        return module.get_function("categorical_accuracy")

    def __call__(self, y_pred, y_targ):
        threads = min(self.model.batch_size, 1024)
        blocks = max(self.model.batch_size, 1024) // threads + 1
        self.kernel(y_targ, y_pred, self.cost,
                    np.int32(self.model.batch_size), np.int32(self.shape[1]),
                    grid=(blocks, 1, 1), block=(threads, 1, 1),
                    stream=self.model.stream)
        return gpuarray.sum(self.cost).get() * 100 / self.model.batch_size
