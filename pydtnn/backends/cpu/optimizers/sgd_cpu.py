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
from pydtnn.optimizers import SGD

from pydtnn.backends.cpu.layers import LayerCPU
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from pydtnn.optimizers import Layer_types
else: Layer_types = None

class SGDCPU(OptimizerCPU, SGD):

    def initialize(self, list_layers: list[Layer_types]) -> None:

        for layer in list_layers:
            list_grad_vars = list(layer.grad_vars.keys())
                    
            if len(list_grad_vars) != 0:
                self.context[layer] = dict[str, np.ndarray]()
                for w_ in list_grad_vars:
                    w = getattr(layer, w_)
                    self.context[layer]["velocity_%s" % w_] = np.zeros_like(w, dtype=layer.model.dtype)

    def update(self, layer: LayerCPU) -> None:
        for w_, dw_ in layer.grad_vars.items():
            w, dw = getattr(layer, w_), getattr(layer, dw_)
            velocity: np.ndarray = self.context[layer]["velocity_%s" % w_]
            w: np.ndarray
            dw: np.ndarray
            # NOTE: The operations are unrolled in order to reduce the memory consumed by intermediate copies of the variables during the operations.
            
            # velocity = self.momentum * velocity + dw
            # NOTE/ Future FIXME: This will raise an error if the model is working in "int8" due is trying to assing a float64 value into a int8 ndarray.
            velocity *= self.momentum
            velocity += dw
            
            #if self.nesterov:
            #    w -= self.learning_rate * (self.decay * w + dw + self.momentum * velocity)
            #else:
            #    w -= self.learning_rate * (self.decay * w + velocity)
            if self.nesterov:
                v = velocity * self.momentum
                v += dw
            else:
                v = velocity
            _w = w * self.decay
            _w += v
            _w *= self.learning_rate
            w -= _w

            # TODO: check if "del" worths to reduce the memory without increasing the execution time.
            del v
            del _w
