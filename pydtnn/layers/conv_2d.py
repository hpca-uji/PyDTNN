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
import numpy as np

class Conv2D(Layer, ABC):

    def __init__(self, nfilters=1, filter_shape=(3, 3), grouping=None, padding=0, stride=1,
                 activation="", use_bias=True, weights_initializer="glorot_uniform",
                 biases_initializer="zeros"):
        super().__init__()
        self.co = nfilters
        self.filter_shape = filter_shape
        self.grouping = grouping
        self.padding = padding
        self.stride = stride
        self.vpadding, self.hpadding = (padding, padding) if isinstance(padding, int) else padding
        self.vstride, self.hstride = (stride, stride) if isinstance(stride, int) else stride
        self.act = getattr(activations, activation, None)
        self.use_bias = use_bias
        self.weights_initializer = getattr(initializers, weights_initializer)
        self.biases_initializer = getattr(initializers, biases_initializer)
        self.grad_vars = {"weights": "dw"}
        if self.use_bias:
            self.grad_vars["biases"] = "db"
        self.debug = False
        # The next attributes will be initialized later
        self.ci = self.hi = self.wi = self.kh = self.kw = self.ho = self.wo = 0
        self.weights_shape = None
        # @warning: do not do this (affects the gpu version) self.forward = self.backward = None

    def initialize(self, prev_shape, need_dx=True):
        super().initialize(prev_shape, need_dx)
        self.hi, self.wi, self.ci = prev_shape
        self.kh, self.kw = self.filter_shape
        if self.grouping == "depthwise":
            self.co = self.ci
            self.weights_shape = (self.ci, *self.filter_shape)
        elif self.grouping == "pointwise":
            self.kh = self.kw = 1
            self.weights_shape = (self.ci, self.co)
        else:
            self.weights_shape = (self.ci, *self.filter_shape, self.co)
        self.ho = (self.hi + 2 * self.vpadding - self.kh) // self.vstride + 1
        self.wo = (self.wi + 2 * self.hpadding - self.kw) // self.hstride + 1
        self.shape = (self.ho, self.wo, self.co)
        self.nparams = np.prod(self.weights_shape) + (self.co if self.use_bias else 0)

    def show(self, attrs=""):
        super().show("|{:^19s}|{:^24s}|".format(str(self.weights.shape),
                                                f"padd=({self.vpadding},{self.hpadding}), "
                                                f"stride=({self.vstride},{self.hstride})"))
