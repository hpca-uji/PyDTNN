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


class Nadam(Optimizer):
    """
    Nadam optimizer
    """

    def __init__(self, learning_rate=1e-2, beta1=0.99, beta2=0.999, epsilon=1e-7, decay=0.0, dtype=np.float32):
        super().__init__()
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.decay = decay
        self.dtype = dtype

    def update(self, layer):
        lr = self.learning_rate
        it = getattr(layer, "it", 0) + 1
        setattr(layer, "it", it)

        for w_, dw_ in layer.grad_vars.items():
            w, dw = getattr(layer, w_), getattr(layer, dw_)
            m = getattr(layer, "m_%s" % w_, np.zeros_like(w, dtype=layer.model.dtype))
            v = getattr(layer, "v_%s" % w_, np.zeros_like(w, dtype=layer.model.dtype))

            m = self.beta1 * m + (1 - self.beta1) * dw
            v = self.beta2 * v + (1 - self.beta2) * dw ** 2

            mt = (m + (1 - self.beta1) * dw) / (1 - self.beta1 ** it)
            vt = v / (1 - self.beta2 ** it)

            w -= lr * (self.decay * w + (mt / np.sqrt(vt + self.epsilon)))

            setattr(layer, w_, w)
            setattr(layer, "m_%s" % w_, m)
            setattr(layer, "v_%s" % w_, v)
