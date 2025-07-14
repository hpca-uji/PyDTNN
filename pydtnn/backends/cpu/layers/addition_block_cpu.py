#
#  This file is part of Python Distributed Training of Neural Networks (PyDTNN)
#
#  Copyright (C) 2021-25 Universitat Jaume I
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
from pydtnn.cython_modules import eltw_sum_cython
from pydtnn.tracers import PYDTNN_MDL_EVENT, PYDTNN_MDL_EVENTS, PYDTNN_OPS_EVENT, PYDTNN_OPS_EVENTS, \
    PYDTNN_EVENT_FINISHED, PYDTNN_MDL_EVENT_enum, PYDTNN_OPS_EVENT_enum    
import numpy as np


class AdditionBlockCPU(AbstractBlockLayerCPU, AdditionBlock):

    def initialize_block_layer(self):
        super().initialize_block_layer()
        assert all([o == self.out_shapes[0] for o in self.out_shapes])
        self.shape = self.out_shapes[0]

    def forward(self, x: np.ndarray) -> np.ndarray:
        
        num_paths = len(self.paths)
        p = self.paths[0]
        x_forward = x
        for layer in p:
            self.model.tracer.emit_event(PYDTNN_MDL_EVENT, layer.id * PYDTNN_MDL_EVENTS + PYDTNN_MDL_EVENT_enum.FORWARD)
            x_forward = layer.forward(x_forward)
            self.model.tracer.emit_event(PYDTNN_MDL_EVENT, PYDTNN_EVENT_FINISHED)
        sum_forwards = x_forward

        for i in range(1, num_paths):
            p = self.paths[i]
            x_forward = x
            for layer in p:
                self.model.tracer.emit_event(PYDTNN_MDL_EVENT, layer.id * PYDTNN_MDL_EVENTS + PYDTNN_MDL_EVENT_enum.FORWARD)
                x_forward = layer.forward(x_forward)
                self.model.tracer.emit_event(PYDTNN_MDL_EVENT, PYDTNN_EVENT_FINISHED)
            self.model.tracer.emit_event(PYDTNN_OPS_EVENT, self.id * PYDTNN_OPS_EVENTS + PYDTNN_OPS_EVENT_enum.FORWARD_ELTW_SUM)
            #eltw_sum_cython(sum_forwards.reshape(-1, copy=False), x_forward.reshape(-1, copy=False))
            sum_forwards += x_forward
            self.model.tracer.emit_event(PYDTNN_OPS_EVENT, PYDTNN_EVENT_FINISHED)
        #print(f"sum_forwards\n{sum_forwards}")
        #print(f"sum_forwards2\n{sum_forwards2}")

        return sum_forwards
    # --- END forward --- #

    def backward(self, dy:np.ndarray) -> np.ndarray:
        num_paths = len(self.paths)
        p = self.paths[0]
        dx_backward = dy
        for layer in reversed(p):
            self.model.tracer.emit_event(PYDTNN_MDL_EVENT, layer.id * PYDTNN_MDL_EVENTS + PYDTNN_MDL_EVENT_enum.FORWARD)
            dx_backward = layer.backward(dx_backward)
            self.model.tracer.emit_event(PYDTNN_MDL_EVENT, PYDTNN_EVENT_FINISHED)
        dx = dx_backward

        for i in range(1, num_paths):
            p = self.paths[i]
            dx_backward = dy
            for layer in reversed(p):
                self.model.tracer.emit_event(PYDTNN_MDL_EVENT, layer.id * PYDTNN_MDL_EVENTS + PYDTNN_MDL_EVENT_enum.BACKWARD)
                dx_backward = layer.backward(dx_backward)
                self.model.tracer.emit_event(PYDTNN_MDL_EVENT, PYDTNN_EVENT_FINISHED)
            
            self.model.tracer.emit_event(PYDTNN_OPS_EVENT, self.id * PYDTNN_OPS_EVENTS + PYDTNN_OPS_EVENT_enum.BACKWARD_ELTW_SUM)
            #eltw_sum_cython(dx.reshape(-1, copy=False), dx_backward.reshape(-1, copy=False))
            dx += dx_backward
            self.model.tracer.emit_event(PYDTNN_OPS_EVENT, PYDTNN_EVENT_FINISHED)
        return dx
    # --- END backward --- #