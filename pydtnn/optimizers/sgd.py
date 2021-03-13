#
#  This file is part of Python Distributed Training of Neural Networks (PyDTNN)
# 
#  Copyright (C) 2021 Universitat Jaume I
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

import numpy as np

from .optimizer import Optimizer


class SGD(Optimizer):
    """
    SGD Optimizer
    """

    def __init__(self, learning_rate=1e-2, momentum=0.9, nesterov=False, decay=0.0, dtype=np.float32):
        super().__init__()
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.nesterov = nesterov
        self.decay = decay
        self.dtype = dtype

    def update(self, layer):
        lr = self.learning_rate
        for w_, dw_ in layer.grad_vars.items():
            w, dw = getattr(layer, w_), getattr(layer, dw_)
            velocity = getattr(layer, "velocity_%s" % w_, np.zeros_like(w, dtype=layer.model.dtype))

            velocity = self.momentum * velocity + dw
            if self.nesterov:
                w -= lr * (self.decay * w + dw + self.momentum * velocity)
            else:
                w -= lr * (self.decay * w + velocity)

            setattr(layer, w_, w)
            setattr(layer, "velocity_%s" % w_, velocity)
