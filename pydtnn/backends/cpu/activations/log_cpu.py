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

class LogCPU(ActivationCPU, Log):

    def forward(self, x:ndarray) -> ndarray:
        # return np.log(1 / (1 + np.exp(-x)))
        x *= -1
        np.exp(x, out=x, casting='unsafe', dtype=x.dtype)
        x += 1
        np.reciprocal(x, out=x)
        return np.log(x).astype(dtype=self.model.dtype, copy=False)

    def backward(self, dy:ndarray) -> ndarray:
        # return 1 / (np.exp(dy) + 1)
        np.exp(dy, out=dy, casting='unsafe', dtype=dy.dtype)
        dy += 1
        np.reciprocal(dy, out=dy)
        return dy
