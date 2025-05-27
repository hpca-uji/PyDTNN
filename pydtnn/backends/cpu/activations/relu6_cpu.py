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

from pydtnn.activations.relu6 import Relu6
from pydtnn.backends.cpu.activations.activation_cpu import ActivationCPU
from pydtnn.cython_modules import capped_relu_cython
from pydtnn.model import TRAIN_MODE
from numpy import ndarray


class Relu6CPU(Relu6, ActivationCPU):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.mask: ndarray = None

    def forward(self, x: ndarray) -> ndarray:
        self.y, mask = capped_relu_cython(x, self.cap)
        if self.model.mode == TRAIN_MODE:
            self.mask = mask
        return self.y

    def backward(self, dy: ndarray) -> ndarray | None:
        if self.need_dx:
            return dy * self.mask
