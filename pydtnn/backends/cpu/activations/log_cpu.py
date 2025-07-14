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

from pydtnn.activations.log import Log
from pydtnn.backends.cpu.activations.activation_cpu import ActivationCPU
from numpy import ndarray
from pydtnn.cython_modules import log_fwd_cython, log_bwd_cython 

class LogCPU(ActivationCPU, Log):

    def initialize(self, prev_shape):
        super().initialize(prev_shape)
        self.y = np.empty(shape=(self.model.batch_size, *self.shape), dtype=self.model.dtype)
        self.dx = np.empty(shape=(self.model.batch_size, *self.shape), dtype=self.model.dtype)

    def forward(self, x:ndarray) -> ndarray:
        y = self.y[:x.shape[0], :]
        log_fwd_cython(x.reshape(-1, copy=False), y.reshape(-1, copy=False))
        return y

    def backward(self, dy:ndarray) -> ndarray:
        dx = self.dx[:dy.shape[0], :]
        log_bwd_cython(dy.reshape(-1, copy=False), dx.reshape(-1, copy=False))

        return dx
