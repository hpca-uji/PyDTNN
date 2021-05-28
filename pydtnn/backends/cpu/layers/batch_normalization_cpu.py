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

from pydtnn.cython_modules import bn_inference_cython
from pydtnn.layers import BatchNormalization
from pydtnn.model import EVALUATE_MODE, TRAIN_MODE
from .layer_cpu import LayerCPU

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
            x = x.reshape(-1, self.ci)

        if self.model.mode == TRAIN_MODE:
            n = np.array([x.shape[0]], dtype=self.model.dtype)
            if self.sync_stats and self.model.comm is not None:
                self.model.comm.Allreduce(MPI.IN_PLACE, n, op=MPI.SUM)

            mu = mean(x, n, self.model.comm)
            xc = (x - mu)
            var = mean(xc ** 2, n, self.model.comm)

            self.std = np.sqrt(var + self.epsilon)
            self.xn = xc / self.std
            y = self.gamma * self.xn + self.beta

            self.running_mean = self.momentum * self.running_mean + (1.0 - self.momentum) * mu
            self.running_var = self.momentum * self.running_var + (1.0 - self.momentum) * var
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

        return y

    def backward(self, dy):
        if self.spatial:
            dy = dy.reshape(-1, self.ci)

        n = dy.shape[0]
        self.dgamma = np.sum(dy * self.xn, axis=0)
        self.dbeta = np.sum(dy, axis=0)

        if self.need_dx:
            dx = (self.gamma / (self.std * n)) * (n * dy - self.xn * self.dgamma - self.dbeta)
            dx = dx.astype(self.model.dtype)

            if self.spatial:
                dx = dx.reshape(-1, self.hi, self.wi, self.ci)
            return dx
