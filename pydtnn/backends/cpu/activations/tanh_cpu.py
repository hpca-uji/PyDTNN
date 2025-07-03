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

from pydtnn.activations.tanh import Tanh
from pydtnn.backends.cpu.activations.activation_cpu import ActivationCPU


class TanhCPU(ActivationCPU, Tanh):

    def forward(self, x:np.ndarray) -> np.ndarray:
        self.y = np.tanh(x, casting="unsafe", dtype=self.model.dtype)
        return self.y

    def backward(self, dy:np.ndarray) -> np.ndarray | None:
        if self.need_dx:
            # return 1 - np.tanh(dy) ** 2
            np.tanh(dy, out=dy, casting="unsafe", dtype=dy.dtype)
            dy **= 2
            return 1 - dy
