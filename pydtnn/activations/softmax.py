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

class Softmax(Activation):

    def forward(self, x: ndarray | TensorGPU) -> ndarray | TensorGPU:
        print("TEST")
        self.y = np.exp(x - np.max(x, axis=1, keepdims=True))
        self.y /= np.sum(self.y, axis=1, keepdims=True)
        return self.y

    def backward(self, dy: ndarray | TensorGPU | None) -> ndarray | TensorGPU | None:
        print("TEST2")
        return self.y * (dy - (dy * self.y).sum(axis=1, keepdims=True))
