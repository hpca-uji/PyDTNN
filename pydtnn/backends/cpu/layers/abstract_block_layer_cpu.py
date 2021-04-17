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

from abc import ABC

from pydtnn.backends.cpu.layers import LayerCPU
from pydtnn.layers.abstract_block_layer import AbstractBlockLayer


class AbstractBlockLayerCPU(LayerCPU, AbstractBlockLayer, ABC):

    def initialize_block_layer(self):
        for p in self.paths:
            prev_shape = self.prev_shape
            for i, layer in enumerate(p):
                layer.set_model(self.model)
                layer.initialize(prev_shape, self.need_dx)
                prev_shape = layer.shape
                self.fwd_time += layer.fwd_time
                self.bwd_time += layer.bwd_time
                self.nparams += layer.nparams
            self.out_shapes.append(prev_shape)
