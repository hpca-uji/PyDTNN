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

from pydtnn.activations.softmax import Softmax
from pydtnn.backends.cpu.activations.activation_cpu import ActivationCPU

class SoftmaxCPU(ActivationCPU, Softmax):

    def __init__(self, shape:np.ndarray = (1,)):
        super().__init__(shape)        

    def forward(self, x:np.ndarray) -> np.ndarray:
        #self.y = np.exp(x - np.max(x, axis=1, keepdims=True))
        #self.y /= np.sum(self.y, axis=1, keepdims=True)
        #return self.y
        x -= np.max(x, axis=1, keepdims=True)
        self.y = np.exp(x)
        self.y /= np.sum(self.y, axis=1, keepdims=True)
        self.y = self.y.astype(dtype=self.model.dtype, copy=False)
        return self.y

    def backward(self, dy:np.ndarray) -> np.ndarray | None:        
        #return self.y * (dy - (dy * self.y).sum(axis=1, keepdims=True))
        _dy = (dy * self.y)
        _dy = _dy.sum(axis=1, keepdims=True)
        dy -= _dy
        return self.y * dy
