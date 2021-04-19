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

from abc import ABC

from .layer import Layer
from .. import initializers


class BatchNormalization(Layer, ABC):

    def __init__(self, beta=0.0, gamma=1.0, momentum=0.9, epsilon=1e-5,
                 moving_mean_initializer="zeros", moving_variance_initializer="zeros",
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
