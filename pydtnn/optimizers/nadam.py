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
#  with this program.  If not, see <https://www.gnu.org/licenses/>.
#

from abc import ABC

import numpy as np

from .optimizer import Optimizer


class Nadam(Optimizer, ABC):
    """
    Nadam optimizer
    """

    def __init__(self, learning_rate:float=1e-2, beta1:float=0.99, beta2:float=0.999, 
                 epsilon:float=1e-7, decay:float=0.0, dtype:np.dtype=np.float32):
        super().__init__()
        self.learning_rate:float = learning_rate
        self.beta1:float = beta1
        self.beta2:float = beta2
        self.epsilon:float = epsilon
        self.decay:float = decay
        self.dtype:np.dtype = dtype
