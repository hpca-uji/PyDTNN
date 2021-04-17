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

from pydtnn.layers import AdditionBlock
from pydtnn.tracers import PYDTNN_MDL_EVENT, PYDTNN_MDL_EVENTS, PYDTNN_MDL_FORWARD, PYDTNN_MDL_BACKWARD, \
    PYDTNN_OPS_EVENT, PYDTNN_OPS_EVENTS, PYDTNN_OPS_FORWARD_ELTW_SUM, \
    PYDTNN_OPS_BACKWARD_ELTW_SUM
from . import LayerGPU
from ..libs import libcudnn as cudnn


class AdditionBlockGPU(LayerGPU, AdditionBlock):

    def initialize_block_layer(self):
        super().initialize_block_layer()
        for p_i, p in enumerate(self.paths):
            prev_shape = self.prev_shape
            x = self.x
            for i, layer in enumerate(p):
                layer.set_model(self.model)
                layer.initialize(prev_shape, self.need_dx, x)
                x = layer.y
                if p_i == 0 and (len(p) - 1) == i:
                    self.y = x
                prev_shape = layer.shape
                self.fwd_time += layer.fwd_time
                self.bwd_time += layer.bwd_time
                self.nparams += layer.nparams
            self.out_shapes.append(prev_shape)
        assert all([o == self.out_shapes[0] for o in self.out_shapes])
        self.shape = self.out_shapes[0]

    def forward(self, x):
        for i, p in enumerate(self.paths):
            y_i = x
            for layer in p:
                self.model.tracer.emit_event(PYDTNN_MDL_EVENT, layer.id * PYDTNN_MDL_EVENTS + PYDTNN_MDL_FORWARD)
                y_i = layer.forward(y_i)
                self.model.tracer.emit_event(PYDTNN_MDL_EVENT, 0)
            if i == 0:
                y = y_i
            else:
                alpha, beta = 1.0, 1.0
                self.model.tracer.emit_event(PYDTNN_OPS_EVENT,
                                             self.id * PYDTNN_OPS_EVENTS + PYDTNN_OPS_FORWARD_ELTW_SUM)
                # noinspection PyUnboundLocalVariable
                cudnn.cudnnAddTensor(self.model.cudnn_handle, alpha, y_i.desc,
                                     y_i.ptr, beta, y.desc, y.ptr)
                self.model.tracer.emit_event(PYDTNN_OPS_EVENT, 0)
        return y

    def backward(self, dy):
        for i, p in enumerate(self.paths):
            dx_i = dy
            for layer in reversed(p):
                self.model.tracer.emit_event(PYDTNN_MDL_EVENT, layer.id * PYDTNN_MDL_EVENTS + PYDTNN_MDL_BACKWARD)
                dx_i = layer.backward(dx_i)
                self.model.tracer.emit_event(PYDTNN_MDL_EVENT, 0)
            if i == 0:
                dx = dx_i
            else:
                alpha, beta = 1.0, 1.0
                self.model.tracer.emit_event(PYDTNN_OPS_EVENT,
                                             self.id * PYDTNN_OPS_EVENTS + PYDTNN_OPS_BACKWARD_ELTW_SUM)
                # noinspection PyUnboundLocalVariable
                cudnn.cudnnAddTensor(self.model.cudnn_handle, alpha, dx_i.desc,
                                     dx_i.ptr, beta, dx.desc, dx.ptr)
                self.model.tracer.emit_event(PYDTNN_OPS_EVENT, 0)
        return dx
