"""
PyDTNN Activation layers
"""

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
from ..layers import Layer
from ..model import TRAIN_MODE
from ..cython_modules import relu_cython


class ActivationLayer(Layer):

    def initialize(self, prev_shape, need_dx=True, x=None):
        super().initialize(prev_shape, need_dx, x)
        self.shape = prev_shape


class Sigmoid(ActivationLayer):

    def __init__(self, shape=(1,)):
        super(Sigmoid, self).__init__(shape)
        # The next attributes will be initialized later
        self.y = None

    def forward(self, x):
        self.y = 1 / (1 + np.exp(-x))
        return self.y

    def backward(self, dy):
        if self.need_dx:
            return dy * (self.y * (1 - self.y))


class Relu(ActivationLayer):

    def __init__(self, shape=(1,)):
        super(Relu, self).__init__(shape)
        # The next attributes will be initialized later
        self.mask = None

    def forward(self, x):
        y, mask = relu_cython(x)
        if self.model.mode == TRAIN_MODE:
            self.mask = mask
        return y

    def backward(self, dy):
        if self.need_dx:
            return dy * self.mask


class Tanh(ActivationLayer):

    def __init__(self, shape=(1,)):
        super(Tanh, self).__init__(shape)

    def forward(self, x):
        return np.tanh(x)

    def backward(self, dy):
        if self.need_dx:
            return 1 - np.tanh(dy) ** 2


class Arctanh(ActivationLayer):

    def __init__(self, shape=(1,)):
        super(Arctanh, self).__init__(shape)

    def forward(self, x):
        return np.arctan(x)

    def backward(self, dy):
        if self.need_dx:
            return 1 / (1 + dy ** 2)


class Log(ActivationLayer):

    def __init__(self, shape=(1,)):
        super(Log, self).__init__(shape)

    def forward(self, x):
        return log(1 / (1 + np.exp(-x)))

    def backward(self, dy):
        if self.need_dx:
            return 1 / (np.exp(dy) + 1)


class Softmax(ActivationLayer):

    def __init__(self, shape=(1,)):
        super(Softmax, self).__init__(shape)
        # The next attributes will be initialized later
        self.y = None

    def forward(self, x):
        self.y = np.exp(x - np.max(x, axis=1, keepdims=True))
        self.y /= np.sum(self.y, axis=1, keepdims=True)
        return self.y

    def backward(self, dy):
        if self.need_dx:
            return self.y * (dy - (dy * self.y).sum(axis=1, keepdims=True))


# Aliases
sigmoid = Sigmoid
relu = Relu
tanh = Tanh
arctanh = Arctanh
log = Log
softmax = Softmax
