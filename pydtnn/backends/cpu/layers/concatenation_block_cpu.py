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

import numpy as np

from pydtnn.backends.cpu.layers.abstract_block_layer_cpu import AbstractBlockLayerCPU
from pydtnn.layers import ConcatenationBlock
from pydtnn.tracers import PYDTNN_MDL_EVENT, PYDTNN_MDL_EVENTS, PYDTNN_OPS_EVENT, PYDTNN_OPS_EVENTS, \
    PYDTNN_EVENT_FINISHED, PYDTNN_MDL_EVENT_enum, PYDTNN_OPS_EVENT_enum  
from pydtnn.utils import PYDTNN_TENSOR_FORMAT

CONCAT_DIM_NCHW  = 1
CONCAT_DIM_NHWC = -1

class ConcatenationBlockCPU(AbstractBlockLayerCPU, ConcatenationBlock):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # The next attributes will be initialized later
        self.out_co:list[int] = None
        self.idx_co:np.ndarray = None
        self.concat_dim:int = None

    def initialize_block_layer(self):
        super().initialize_block_layer()        
        if self.model.tensor_format == PYDTNN_TENSOR_FORMAT.NCHW:
            assert all([tuple(o[1:]) == tuple(self.out_shapes[0][1:]) for o in self.out_shapes])
            self.out_co = [s[0] for s in self.out_shapes]
            self.idx_co = np.cumsum(self.out_co, axis=0)
            self.shape = (sum(self.out_co), *self.out_shapes[0][1:])
            self.concat_dim = CONCAT_DIM_NCHW
        else: # Assuming PYDTNN_TENSOR_FORMAT_NHWC
            assert all([tuple(o[:-1]) == tuple(self.out_shapes[0][:-1]) for o in self.out_shapes])
            self.out_co = [s[-1] for s in self.out_shapes]
            self.idx_co:np.ndarray = np.cumsum(self.out_co, axis=0)
            self.shape:tuple[int] = (*self.out_shapes[0][:-1], sum(self.out_co))
            self.concat_dim = CONCAT_DIM_NHWC

    def forward(self, x:np.ndarray) -> np.ndarray:
        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, self.id * PYDTNN_OPS_EVENTS + PYDTNN_OPS_EVENT_enum.FORWARD_REPLICATE)
        _x = [0] * len(self.paths)
        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, PYDTNN_EVENT_FINISHED)

        for i, p in enumerate(self.paths):
            x_forward = x
            for layer in p:
                self.model.tracer.emit_event(PYDTNN_MDL_EVENT, layer.id * PYDTNN_MDL_EVENTS + PYDTNN_MDL_EVENT_enum.FORWARD)
                x_forward = layer.forward(x_forward)
                self.model.tracer.emit_event(PYDTNN_MDL_EVENT, PYDTNN_EVENT_FINISHED)
            _x[i] = x_forward
        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, self.id * PYDTNN_OPS_EVENTS + PYDTNN_OPS_EVENT_enum.FORWARD_CONCAT)
        y = np.concatenate(_x, axis=self.concat_dim)
        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, PYDTNN_EVENT_FINISHED)

        return y

    def backward(self, dy:np.ndarray) -> np.ndarray:
        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, self.id * PYDTNN_OPS_EVENTS + PYDTNN_OPS_EVENT_enum.BACKWARD_SPLIT)
        dx = np.split(dy, self.idx_co[:-1], axis=self.concat_dim)
        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, PYDTNN_EVENT_FINISHED)
        
        num_paths = len(self.paths)

        p = self.paths[0]
        for layer in reversed(p):
                self.model.tracer.emit_event(PYDTNN_MDL_EVENT, layer.id * PYDTNN_MDL_EVENTS + PYDTNN_MDL_EVENT_enum.BACKWARD)
                dx[0] = layer.backward(dx[0])
                self.model.tracer.emit_event(PYDTNN_MDL_EVENT, PYDTNN_EVENT_FINISHED)

        for i in range(1, num_paths):
            p = self.paths[i]
            for layer in reversed(p):
                self.model.tracer.emit_event(PYDTNN_MDL_EVENT, layer.id * PYDTNN_MDL_EVENTS + PYDTNN_MDL_EVENT_enum.BACKWARD)
                dx[i] = layer.backward(dx[i])
                self.model.tracer.emit_event(PYDTNN_MDL_EVENT, PYDTNN_EVENT_FINISHED)

            self.model.tracer.emit_event(PYDTNN_OPS_EVENT, self.id * PYDTNN_OPS_EVENTS + PYDTNN_OPS_EVENT_enum.BACKWARD_ELTW_SUM)
            dx[0] += dx[i]
            self.model.tracer.emit_event(PYDTNN_OPS_EVENT, PYDTNN_EVENT_FINISHED)
        return dx[0]
