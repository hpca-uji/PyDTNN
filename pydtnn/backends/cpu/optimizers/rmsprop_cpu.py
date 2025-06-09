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

from pydtnn.backends.cpu.optimizers import OptimizerCPU
from pydtnn.optimizers import RMSProp

from pydtnn.backends.cpu.layers import LayerCPU
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from pydtnn.optimizers import Layer_types
else: Layer_types = None

class RMSPropCPU(OptimizerCPU, RMSProp):

    def initialize(self, list_layers: list[Layer_types]) -> None:

        for layer in list_layers:
            list_grad_vars = list(layer.grad_vars.keys())
                    
            if len(list_grad_vars) != 0:
                self.context[layer] = dict[str, np.ndarray]()
                for w_ in list_grad_vars:
                    w = getattr(layer, w_)
                    self.context[layer]["cache_%s" % w_] = np.zeros_like(w, dtype=layer.model.dtype)

    def update(self, layer: LayerCPU) -> None:
        for w_, dw_ in layer.grad_vars.items():
            w, dw = getattr(layer, w_), getattr(layer, dw_)
            cache:np.ndarray = self.context[layer]["cache_%s" % w_]
            w:np.ndarray
            dw:np.ndarray

            # NOTE: The operations are unrolled in order to reduce the memory consumed by intermediate copies of the variables during the operations.

            #cache = self.rho * cache + (1 - self.rho) * dw ** 2
            cache *= self.rho
            _dw = dw ** 2
            _dw *= (1 - self.rho)
            cache += _dw
            #w -= self.learning_rate * (self.decay * w + (dw / np.sqrt(cache + self.epsilon)))
            w -= (self.learning_rate * self.decay) * w
            _cache = cache + self.epsilon
            _cache = np.sqrt(_cache)
            _dw = dw / _cache 
            _dw *= self.learning_rate
            w -= _dw

            # TODO: check if "del" worths to reduce the memory without increasing the execution time.
            del _cache
            del _dw
