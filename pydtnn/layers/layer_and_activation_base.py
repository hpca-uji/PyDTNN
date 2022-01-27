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

from abc import ABC, abstractmethod

import numpy as np


class LayerAndActivationBase(ABC):

    def __init__(self, shape=()):
        self.nparams = 0
        self.shape = shape
        self.weights = np.array([])
        self.biases = np.array([])
        self.act = None
        self.grad_vars = {}
        self.fwd_time = np.zeros((4,), dtype=np.float32)
        self.bwd_time = np.zeros((4,), dtype=np.float32)
        self.paths = []
        self.need_dx = True
        self.reqs_allred = {}
        # The next attributes will be initialized later
        self.id = None
        self.model = None
        self.prev_shape = None
        self.is_block_layer = False
        self.stream_2 = None

    @property
    def _id_prefix(self):
        prefix = ''
        if self.id is not None and self.model is not None:
            try:
                model__last_layer = self.model.layers[-1]
            except IndexError:
                max_digits = 1
            else:
                model__last_id = model__last_layer.id
                if len(model__last_layer.children):
                    model__last_id = model__last_layer.children[-1].id
                max_digits = len(str(model__last_id))
            prefix = "{:0{width}d}_".format(self.id, width=max_digits)
        return prefix

    def __repr__(self):
        return f"{self._id_prefix}{type(self).__name__}"

    def set_model(self, parent_model):
        self.model = parent_model
        self.id = next(self.model.layer_id)

    def initialize(self, prev_shape, need_dx=True):
        self.prev_shape = prev_shape
        self.need_dx = need_dx

    @abstractmethod
    def forward(self, x):
        pass

    @abstractmethod
    def backward(self, dy):
        pass

    @abstractmethod
    def reduce_weights_async(self):
        pass

    @abstractmethod
    def wait_allreduce_async(self):
        pass

    @abstractmethod
    def reduce_weights_sync(self):
        pass

    def show(self, attrs=""):
        if not attrs:
            attrs = "|{:19s}|{:^37s}|".format("", "")
        print(f"|{self.id:^7d}|{type(self).__name__:^26s}|{self.nparams:^9d}|{str(self.shape):^15}" + attrs)

    @property
    def children(self):
        children = []
        for path in self.paths:
            children += [layer for layer in path]
        return children

    def update_weights(self, optimizer):
        optimizer.update(self)
