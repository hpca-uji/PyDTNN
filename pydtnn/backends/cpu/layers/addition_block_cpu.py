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

from pydtnn.backends.cpu.layers.abstract_block_layer_cpu import AbstractBlockLayerCPU
from pydtnn.layers import AdditionBlock
from pydtnn.tracers import PYDTNN_OPS_EVENT, PYDTNN_OPS_EVENTS, PYDTNN_MDL_FORWARD, PYDTNN_MDL_BACKWARD, \
    PYDTNN_MDL_EVENT, PYDTNN_MDL_EVENTS, PYDTNN_OPS_BACKWARD_ELTW_SUM, PYDTNN_OPS_FORWARD_ELTW_SUM


class AdditionBlockCPU(AbstractBlockLayerCPU, AdditionBlock):

    def initialize_block_layer(self):
        super().initialize_block_layer()
        assert all([o == self.out_shapes[0] for o in self.out_shapes])
        self.shape = self.out_shapes[0]

    def forward(self, x):
        x = [x] * len(self.paths)
        for i, p in enumerate(self.paths):
            for layer in p:
                self.model.tracer.emit_event(PYDTNN_MDL_EVENT, layer.id * PYDTNN_MDL_EVENTS + PYDTNN_MDL_FORWARD)
                x[i] = layer.forward(x[i])
                self.model.tracer.emit_event(PYDTNN_MDL_EVENT, 0)
            if i > 0:
                self.model.tracer.emit_event(PYDTNN_OPS_EVENT,
                                             self.id * PYDTNN_OPS_EVENTS + PYDTNN_OPS_FORWARD_ELTW_SUM)
                x[0] += x[i]
                self.model.tracer.emit_event(PYDTNN_OPS_EVENT, 0)
        return x[0]

    def backward(self, dy):
        dx = [dy] * len(self.paths)
        for i, p in enumerate(self.paths):
            for layer in reversed(p):
                self.model.tracer.emit_event(PYDTNN_MDL_EVENT, layer.id * PYDTNN_MDL_EVENTS + PYDTNN_MDL_BACKWARD)
                dx[i] = layer.backward(dx[i])
                self.model.tracer.emit_event(PYDTNN_MDL_EVENT, 0)
            if i > 0:
                self.model.tracer.emit_event(PYDTNN_OPS_EVENT,
                                             self.id * PYDTNN_OPS_EVENTS + PYDTNN_OPS_BACKWARD_ELTW_SUM)
                dx[0] += dx[i]
                self.model.tracer.emit_event(PYDTNN_OPS_EVENT, 0)
        return dx[0]
