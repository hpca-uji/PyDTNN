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

from pydtnn.activations.arctanh import Arctanh
from .activation_cpu import ActivationCPU
from numpy import ndarray

class ArctanhCPU(ActivationCPU, Arctanh):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)        

    def forward(self, x: ndarray) -> ndarray:
        #self.y = np.arctan(x)
        self.y = np.arctan(x, casting="unsafe", dtype=x.dtype)
        return self.y

    def backward(self, dy: ndarray) -> ndarray:
        # return 1 / (1 + dy ** 2)
        dy **= 2
        dy += 1
        np.reciprocal(dy, out=dy)
        return dy
