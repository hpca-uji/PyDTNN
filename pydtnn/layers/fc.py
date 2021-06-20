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

from abc import ABC

from .layer import Layer
from .. import activations
from .. import initializers


class FC(Layer, ABC):

    def __init__(self, shape=(1,), activation="", use_bias=True,
                 weights_initializer="glorot_uniform",
                 biases_initializer="zeros"):
        super().__init__(shape)
        self.act = getattr(activations, activation, None)
        self.use_bias = use_bias
        self.weights_initializer = getattr(initializers, weights_initializer)
        self.biases_initializer = getattr(initializers, biases_initializer)
        self.grad_vars = {"weights": "dw"}
        if self.use_bias:
            self.grad_vars["biases"] = "db"

    def show(self, attrs=""):
        super().show("|{:^19s}|{:^37s}|".format(str(self.weights.shape), ""))
