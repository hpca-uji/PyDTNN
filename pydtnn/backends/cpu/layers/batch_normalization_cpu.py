
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

from pydtnn.cython_modules import bn_inference_cython, bn_training_fwd_cython, \
                                  bn_training_bwd_cython
from pydtnn.layers import BatchNormalization
from pydtnn.model import EVALUATE_MODE, TRAIN_MODE
from pydtnn.utils.best_transpose_0231 import best_transpose_0231
from pydtnn.utils.best_transpose_0312 import best_transpose_0312
from .layer_cpu import LayerCPU
from pydtnn.utils import PYDTNN_TENSOR_FORMAT_NCHW

try:
    # noinspection PyUnresolvedReferences
    from mpi4py import MPI
except (ImportError, ModuleNotFoundError):
    pass


class BatchNormalizationCPU(LayerCPU, BatchNormalization):

    def forward(self, x):

        def mean(data, total, comm):
            if self.sync_stats and comm is not None:
                _mean = np.sum(data, axis=0) / total
                comm.Allreduce(MPI.IN_PLACE, _mean, op=MPI.SUM)
            else:
                _mean = np.mean(data, axis=0)
            return _mean

        if self.spatial:
            if self.model.tensor_format == PYDTNN_TENSOR_FORMAT_NCHW:
                x = best_transpose_0231(x)
            x = x.reshape(-1, self.ci)

        if self.model.mode == TRAIN_MODE:
            if self.sync_stats and self.model.comm is not None:
                n = self.model.nprocs * self.model.batch_size
                # n = np.array([x.shape[0]], dtype=self.model.dtype)
                # self.model.comm.Allreduce(MPI.IN_PLACE, n, op=MPI.SUM)
            else: 
                n = None

            mu = mean(x, n, self.model.comm)
            xc = (x - mu)
            var = mean(xc ** 2, n, self.model.comm)

            self.std = np.sqrt(var + self.epsilon)
            self.xn = xc / self.std
            y = self.gamma * self.xn + self.beta

            self.running_mean = self.momentum * self.running_mean + (1.0 - self.momentum) * mu
            self.running_var = self.momentum * self.running_var + (1.0 - self.momentum) * var

            # y, self.std, self.xn = bn_training_fwd_cython(x, self.gamma, self.beta, \
            #                                               self.running_mean, self.running_var, \
            #                                               self.momentum, self.epsilon)

            self.updated_running_var = True

        elif self.model.mode == EVALUATE_MODE:
            # Original numpy-based code
            # std = np.sqrt(self.running_var + self.epsilon)
            # xn = (x - self.running_mean) / std
            # y = self.gamma * xn + self.beta

            # If self.running_var was updated on training we need to recompute self.inv_std!

            if self.updated_running_var:
                self.updated_running_var = False
                self.inv_std = 1.0 / np.sqrt(self.running_var + self.epsilon)

            y = bn_inference_cython(x, self.running_mean, self.inv_std, self.gamma, self.beta)

        else:
            raise ValueError("Unexpected model mode")

        if self.spatial:
            y = y.reshape(-1, self.hi, self.wi, self.ci)
            if self.model.tensor_format == PYDTNN_TENSOR_FORMAT_NCHW:
                y = best_transpose_0312(y)

        return y

    def backward(self, dy):
        if self.spatial:
            if self.model.tensor_format == PYDTNN_TENSOR_FORMAT_NCHW:
                dy = best_transpose_0231(dy)
            dy = dy.reshape(-1, self.ci)

        n = dy.shape[0]
        self.dgamma = np.sum(dy * self.xn, axis=0)
        self.dbeta = np.sum(dy, axis=0)

        if self.need_dx:
            # dx = (self.gamma / (self.std * n)) * (n * dy - self.xn * self.dgamma - self.dbeta)
            # dx = dx.astype(self.model.dtype)

            dx = bn_training_bwd_cython(dy, self.std, self.xn, self.gamma, self.dgamma, self.dbeta)

            if self.spatial:
                dx = dx.reshape(-1, self.hi, self.wi, self.ci)
                if self.model.tensor_format == PYDTNN_TENSOR_FORMAT_NCHW:
                    dx = best_transpose_0312(dx)

            return dx
