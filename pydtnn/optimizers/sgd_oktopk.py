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
#  with this program.  If not, see <https://www.gnu.org/licenses/>.
#

from abc import ABC

import numpy as np

from .optimizer import Optimizer


class SGD_OkTopk(Optimizer, ABC):
    """
    SGD Ok-Topk Optimizer
    """

    def __init__(self, learning_rate=1e-2, momentum=0.9, nesterov=False, decay=0.0, dtype=np.float32, nprocs=1, comm=None, rank=0, k=10):
        super().__init__()
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.nesterov = nesterov
        self.nprocs = nprocs
        self.residuals = {} 
        self.decay = decay
        self.dtype = dtype
        self.comm = comm
        self.rank = rank
        self.k = k
        # This will be initialized after, it is used on the update:
        self.local_th = None
        self.global_th = None
        self.boundaries = None
