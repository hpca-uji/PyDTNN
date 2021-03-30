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

from ..cython_modules import bn_inference_cython
from ..model import EVALUATE_MODE
from .layer import Layer
from .. import initializers
from ..model import TRAIN_MODE
from ..performance_models import *

try:
    # noinspection PyUnresolvedReferences
    from mpi4py import MPI
except (ImportError, ModuleNotFoundError):
    pass


class BatchNormalization(Layer):

    def __init__(self, beta=0.0, gamma=1.0,
                 momentum=0.9, epsilon=1e-5,
                 moving_mean_initializer="zeros",
                 moving_variance_initializer="ones",
                 sync_stats=False):
        super().__init__()
        self.gamma_init_val = gamma
        self.beta_init_val = beta
        self.momentum = momentum
        self.epsilon = epsilon
        self.moving_mean_initializer = getattr(initializers, moving_mean_initializer)
        self.moving_variance_initializer = getattr(initializers, moving_variance_initializer)
        self.grad_vars = {"beta": "dbeta", "gamma": "dgamma"}
        self.sync_stats = sync_stats
        # The next attributes will be initialized later
        self.spatial = self.co = self.ci = self.hi = self.wi = 0
        self.gamma = self.beta = self.running_mean = self.running_var = None
        self.std = self.xn = None
        self.dgamma = self.dbeta = None
        self.updated_running_var = False
        self.inv_std = None

    def initialize(self, prev_shape, need_dx=True):
        super().initialize(prev_shape, need_dx)
        self.shape = shape_ = prev_shape
        self.spatial = len(self.shape) > 2
        if self.spatial:
            self.co = self.ci = self.shape[0]
            self.hi, self.wi = self.shape[1], self.shape[2]
            shape_ = (self.ci,)
        self.gamma = np.full(shape_, self.gamma_init_val, self.model.dtype)
        self.beta = np.full(shape_, self.beta_init_val, self.model.dtype)
        self.running_mean = self.moving_mean_initializer(shape_, self.model.dtype)
        self.running_var = self.moving_variance_initializer(shape_, self.model.dtype)
        self.inv_std = 1.0 / np.sqrt(self.running_var + self.epsilon)
        self.nparams = self.gamma.size + self.beta.size + self.running_mean.size + self.running_var.size

    def forward(self, x):

        def mean(data, total, comm):
            if self.sync_stats and comm is not None:
                _mean = np.sum(data, axis=0) / total
                comm.Allreduce(MPI.IN_PLACE, _mean, op=MPI.SUM)
            else:
                _mean = np.mean(data, axis=0)
            return _mean

        if self.spatial:
            x = x.transpose(0, 2, 3, 1).reshape(-1, self.ci)

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
            y = y.reshape(-1, self.hi, self.wi, self.ci).transpose(0, 3, 1, 2)

        return y

    def backward(self, dy):
        if self.spatial:
            dy = dy.transpose(0, 2, 3, 1).reshape(-1, self.ci)

        n = dy.shape[0]
        self.dgamma = np.sum(dy * self.xn, axis=0)
        self.dbeta = np.sum(dy, axis=0)

        if self.need_dx:
            dx = (self.gamma / (self.std * n)) * (n * dy - self.xn * self.dgamma - self.dbeta)
            dx = dx.astype(self.model.dtype)

            if self.spatial:
                dx = dx.reshape(-1, self.hi, self.wi, self.ci).transpose(0, 3, 1, 2)
            return dx
