#
#  This file is part of Python Distributed Training of Neural Networks (PyDTNN)
#
#  Copyright (C) 2021-22 Universitat Jaume I
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
from pydtnn.model import TRAIN_MODE

from numpy import ndarray

class LeakyReluCPU(ActivationCPU, LeakyRelu):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.mask: ndarray = None

    def forward(self, x: ndarray) -> ndarray:
        self.y, self.mask = leaky_relu_cython(x, self.negative_slope)
        return self.y

    def backward(self, dy: ndarray) -> ndarray | None:
        if self.need_dx:
            # return dy * self.mask
            dy *= self.mask
            return dy
