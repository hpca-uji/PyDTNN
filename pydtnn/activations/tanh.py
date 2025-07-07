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

import numpy as np

from .activation import Activation

from ..backends.gpu.tensor_gpu import TensorGPU
from numpy import ndarray

class Tanh(Activation):

    def initialize(self, prev_shape):
        super().initialize(prev_shape)
        self.out = np.empty(shape=self.shape)

    def forward(self, x: ndarray | TensorGPU) -> ndarray | TensorGPU:
        np.tanh(x, out=self.out)
        return self.out

    def backward(self, dy: ndarray | TensorGPU | None) -> ndarray | TensorGPU | None:
        #return 1 - np.tanh(dy) ** 2
        np.tanh(dy, self.out)
        self.out **= 2
        return 1 - self.out
