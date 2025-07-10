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

from pydtnn.activations.relu import Relu
from pydtnn.backends.cpu.activations.activation_cpu import ActivationCPU
from pydtnn.cython_modules import relu_cython
import numpy as np

class ReluCPU(ActivationCPU, Relu):

    def __init__(self, shape:tuple[int, ...]=(1,)):
        super().__init__(shape)
        self.mask:np.ndarray = None

    def initialize(self, prev_shape):
        super().initialize(prev_shape)
        self._y = np.empty((self.model.batch_size, *self.prev_shape), dtype=self.model.dtype)
        self._mask = np.empty((self.model.batch_size, *self.prev_shape), dtype=np.int8)

    def forward(self, x:np.ndarray) -> np.ndarray:
        self.y = self._y[:x.shape[0], :]
        self.mask = self._mask[:x.shape[0], :]
        relu_cython(x.reshape(-1, copy=False), self.y.reshape(-1, copy=False), self.mask.reshape(-1, copy=False))
        return self.y

    def backward(self, dy:np.ndarray) -> np.ndarray:
        #return dy * self.mask
        dy *= self.mask
        return dy
