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

from .layer import Layer
from activations import Activation
from initializers import InitializerFunc, glorot_uniform, zeros


class FC(Layer, ABC):

    def __init__(self, shape: tuple[int,...] = (1,), 
                 activation: Activation | None = None, 
                 use_bias=True,
                 weights_initializer: InitializerFunc = glorot_uniform,
                 biases_initializer: InitializerFunc = zeros):
        super().__init__(shape)
        self.act = activation
        self.use_bias = use_bias
        self.weights_initializer = weights_initializer
        self.biases_initializer = biases_initializer
        self.grad_vars = {"weights": "dw"}
        if self.use_bias:
            self.grad_vars["biases"] = "db"

    def show(self, attrs="") -> None:
        super().show("|{:^19s}|{:^37s}|".format(str(self.weights.shape), ""))
