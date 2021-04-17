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
#  with this program. If not, see <https://www.gnu.org/licenses/>.
#

from abc import ABC, abstractmethod

from pydtnn.layers.layer import Layer


class AbstractBlockLayer(Layer, ABC):

    def __init__(self, *args, **kwargs):
        super().__init__(**kwargs)
        self.paths = []
        for p in args:
            self.paths.append(p)
        self.is_block_layer = True
        self.out_shapes = []

    def initialize(self, prev_shape, need_dx=True):
        super().initialize(prev_shape, need_dx)
        self.initialize_block_layer()

    @abstractmethod
    def initialize_block_layer(self):
        pass

    def update_weights(self, optimizer):
        for p in self.paths:
            for layer in p:
                layer.update_weights(optimizer)

    def reduce_weights_async(self):
        for p in self.paths:
            for layer in p:
                layer.reduce_weights_async()

    def wait_allreduce_async(self):
        for p in self.paths:
            for layer in p:
                layer.wait_allreduce_async()

    def reduce_weights_sync(self):
        for p in self.paths:
            for layer in p:
                layer.reduce_weights_sync()
