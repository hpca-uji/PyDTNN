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

from pydtnn.activations.leaky_relu import LeakyRelu
from pydtnn.backends.cpu.activations.activation_cpu import ActivationCPU
from pydtnn.cython_modules import leaky_relu_cython

import numpy as np


class LeakyReluCPU(ActivationCPU, LeakyRelu):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def initialize(self, prev_shape, x = None):
        super().initialize(prev_shape, x)
        self._y = np.empty((self.model.batch_size, *self.prev_shape), dtype=self.model.dtype, order="C")
        self._mask = np.empty((self.model.batch_size, *self.prev_shape), dtype=self.model.dtype, order="C")

    def forward(self, x: np.ndarray) -> np.ndarray:
        self.y = self._y[:x.shape[0], :]
        self.mask = self._mask[:x.shape[0], :]

        leaky_relu_cython(x.reshape(-1, order="C", copy=False), self.y.reshape(-1, order="C", copy=False), self.mask.reshape(-1, order="C", copy=False), self.negative_slope)
        return self.y

    def backward(self, dy: np.ndarray) -> np.ndarray:
        # return dy * self.mask
        np.multiply(dy, self.mask, out=dy, order="C")
        return dy
