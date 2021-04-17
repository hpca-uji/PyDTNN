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

from pydtnn.backends.cpu.layers import LayerCPU
from pydtnn.layers import Flatten
from pydtnn.tracers import PYDTNN_OPS_EVENT, PYDTNN_OPS_EVENTS, PYDTNN_OPS_FORWARD_RESHAPE_Y, \
    PYDTNN_OPS_BACKWARD_RESHAPE_DX


class FlattenCPU(LayerCPU, Flatten):

    def forward(self, x):
        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, self.id * PYDTNN_OPS_EVENTS + PYDTNN_OPS_FORWARD_RESHAPE_Y)
        y = x.reshape((x.shape[0], *self.shape))
        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, 0)
        return y

    def backward(self, dy):
        if self.need_dx:
            self.model.tracer.emit_event(PYDTNN_OPS_EVENT, self.id * PYDTNN_OPS_EVENTS + PYDTNN_OPS_BACKWARD_RESHAPE_DX)
            dx = dy.reshape((dy.shape[0], *self.prev_shape))
            self.model.tracer.emit_event(PYDTNN_OPS_EVENT, 0)
            return dx
