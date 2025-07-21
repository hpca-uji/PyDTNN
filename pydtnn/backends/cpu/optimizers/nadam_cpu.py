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
#  with this program. If not, see <https://www.gnu.org/licenses/>.
#

import numpy as np

from pydtnn.backends.cpu.optimizers import OptimizerCPU
from pydtnn.optimizers import Nadam
from pydtnn.backends.cpu.layers import LayerCPU

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from pydtnn.optimizers import Layer_types
else: Layer_types = None

class NadamCPU(OptimizerCPU, Nadam):

    def initialize(self, list_layers: list[Layer_types]) -> None:

        for layer in list_layers:
            self.context[layer] = dict[str, int | np.ndarray]()
            self.context[layer]["it"] = 0

            for w_ in layer.grad_vars.keys():
                w:np.ndarray = getattr(layer, w_)
                self.context[layer]["m_%s" % w_] = np.zeros_like(w, dtype=layer.model.dtype)
                self.context[layer]["v_%s" % w_] = np.zeros_like(w, dtype=layer.model.dtype)

    def update(self, layer: LayerCPU) -> None:
        self.context[layer]["it"] += 1
        it:int = self.context[layer]["it"]

        for w_, dw_ in layer.grad_vars.items():
            w, dw = getattr(layer, w_), getattr(layer, dw_)
            w: np.ndarray
            dw: np.ndarray
            # Momentum of the weight or bias of the given layer
            m:np.ndarray = self.context[layer]["m_%s" % w_]
            # Velocity of the weight or bias of the given layer
            v:np.ndarray = self.context[layer]["v_%s" % w_]            

            if self.are_all_zeros(w) and self.are_all_zeros(dw) or self.are_all_zeros(m) or self.are_all_zeros(v):
                continue
            else:
                # NOTE: The operations are unrolled in order to reduce the memory consumed by intermediate copies of the variables during the operations.
                #m = self.beta1 * m + (1 - self.beta1) * dw
                m *= self.beta1
                _dw:np.ndarray = (1 - self.beta1) * dw
                m += _dw
                #v = self.beta2 * v + (1 - self.beta2) * dw ** 2
                v *= self.beta2
                _dw = dw ** 2
                _dw *= (1 - self.beta2) 
                v += _dw

                #mt = (m + (1 - self.beta1) * dw) / (1 - self.beta1 ** it)
                mt = (1 - self.beta1) * dw
                mt /= (1 - self.beta1 ** it)
                mt += m

                #vt = v / (1 - self.beta2 ** it)
                vt = v / (1 - self.beta2 ** it)

                #w -= self.learning_rate * (self.decay * w + (mt / np.sqrt(vt + epsilon)))
                w -= (self.learning_rate * self.decay) * w
                vt += self.epsilon
                np.sqrt(vt, out=vt)
                mt /= vt
                mt *= self.learning_rate
                w -= mt
