#
#  This file is part of Python Distributed Training of Neural Networks (PyDTNN)
#
#  Copyright (C) 2025 Universitat Jaume I
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

from pydtnn.activations.relu6 import Relu6
from pydtnn.backends.cpu.activations.activation_cpu import ActivationCPU
from pydtnn.cython_modules import capped_relu_cython
import numpy as np


class Relu6CPU(Relu6, ActivationCPU):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def initialize(self, prev_shape, x = None):
        super().initialize(prev_shape, x)
        self._y = np.empty((self.model.batch_size, *self.prev_shape), dtype=self.model.dtype, order="C")
        self._mask = np.empty((self.model.batch_size, *self.prev_shape), dtype=np.int8, order="C")

    def forward(self, x: np.ndarray) -> np.ndarray:
        self.y: np.ndarray = self._y[:x.shape[0], :]
        self.mask: np.ndarray = self._mask[:x.shape[0], :]
        capped_relu_cython(x.reshape(-1, copy=False, order="C"), 
                           self.y.reshape(-1, copy=False, order="C"), 
                           self.mask.reshape(-1, copy=False, order="C"), 
                           self.cap)
        return self.y

    def backward(self, dy: np.ndarray) -> np.ndarray:
        # return dy * self.mask
        np.multiply(dy, self.mask, out=dy, dtype=self.model.dtype, order="C")
        return dy
