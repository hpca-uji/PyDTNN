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

from pydtnn.activations.sigmoid import Sigmoid
from pydtnn.backends.cpu.activations.activation_cpu import ActivationCPU
from pydtnn.cython_modules import sigmoid_fwd_cython, sigmoid_bwd_cython

class SigmoidCPU(ActivationCPU, Sigmoid):

    def initialize(self, prev_shape):
        super().initialize(prev_shape)
        self.y:np.ndarray = None

        self._y = np.ndarray(shape=(self.model.batch_size, *prev_shape), dtype=self.model.dtype)
        self.dx = np.ndarray(shape=(self.model.batch_size, *prev_shape), dtype=self.model.dtype)

    def forward(self, x:np.ndarray) -> np.ndarray:

        self.y = self._y[:x.shape[0], :]
        sigmoid_fwd_cython(x.reshape(-1, copy=False), self.y.reshape(-1, copy=False))
        return self.y

    def backward(self, dy:np.ndarray) -> np.ndarray:

        dx = self.dx[:dy.shape[0], :]
        sigmoid_bwd_cython(dy.reshape(-1, copy=False), self.y.reshape(-1, copy=False), dx.reshape(-1, copy=False))

        return dx
