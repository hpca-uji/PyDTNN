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

from pydtnn.backends.cpu.layers import LayerCPU
from pydtnn.layers import FC
from pydtnn.model import TRAIN_MODE
from pydtnn.performance_models import matmul_time
from pydtnn.tracers import PYDTNN_OPS_EVENT, PYDTNN_OPS_EVENTS, PYDTNN_EVENT_FINISHED, PYDTNN_OPS_EVENT_enum

class FCCPU(LayerCPU, FC):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.x: np.ndarray = None
        self.dw: np.ndarray = None
        self.db: np.ndarray = None

    def initialize(self, prev_shape, need_dx=True):
        super().initialize(prev_shape, need_dx)
        self.weights = self.weights_initializer((*prev_shape, *self.shape), self.model.dtype)
        # Initialize outputs:
        self.db = np.empty(self.shape, dtype=self.model.dtype)

        if self.use_bias:
            self.biases = self.biases_initializer(self.shape, self.model.dtype)
        self.nparams = self.weights.size + (self.biases.size if self.use_bias else 0)
        # Performance model
        self.fwd_time = \
            matmul_time(m=self.model.batch_size, n=self.weights.shape[1], k=self.weights.shape[0],
                        cpu_speed=self.model.cpu_speed, memory_bw=self.model.memory_bw,
                        dtype=self.model.dtype)
        self.bwd_time = \
            matmul_time(m=self.weights.shape[0], n=self.weights.shape[1], k=self.model.batch_size,
                        cpu_speed=self.model.cpu_speed, memory_bw=self.model.memory_bw,
                        dtype=self.model.dtype) + \
            matmul_time(m=self.model.batch_size, n=self.weights.shape[0], k=self.weights.shape[1],
                        cpu_speed=self.model.cpu_speed, memory_bw=self.model.memory_bw,
                        dtype=self.model.dtype) if need_dx else 0

    def forward(self, x: np.ndarray) -> np.ndarray:
        if self.model.mode == TRAIN_MODE:
            self.x = x
        #_x = x.reshape((-1, np.prod(self.prev_shape)), copy=False)
        #_w = self.weights.reshape(np.prod(self.prev_shape),-1)
        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, self.id * PYDTNN_OPS_EVENTS + PYDTNN_OPS_EVENT_enum.FORWARD_MATMUL)
        #_res:np.ndarray = np.matmul(x, self.weights)
        #_res:np.ndarray = np.matmul(_x, _w)
        # np.matmul(_x, _w, out=self._res)
        res = np.linalg.matmul(x, self.weights)

        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, PYDTNN_EVENT_FINISHED)

        if self.use_bias:
            #_res += self.biases
            res += self.biases

        #return _res
        return res

    def backward(self, dy: np.ndarray) -> np.ndarray | None:
        # self.model.mode = TRAIN_MODE is asumed from this point.
        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, self.id * PYDTNN_OPS_EVENTS + PYDTNN_OPS_EVENT_enum.COMP_DW_MATMUL)
        #self.dw = np.matmul(self.x.T, dy)
        self.dw = np.linalg.matmul(self.x.T, dy)
        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, PYDTNN_EVENT_FINISHED)

        if self.use_bias:
            #self.db = np.sum(dy, axis=0)
            np.sum(dy, axis=0, out=self.db)

        if self.need_dx:
            self.model.tracer.emit_event(PYDTNN_OPS_EVENT, self.id * PYDTNN_OPS_EVENTS + PYDTNN_OPS_EVENT_enum.COMP_DX_MATMUL)
            #dx = np.matmul(dy, self.weights.T)
            #dx = np.linalg.matmul(dy, self.weights.T.reshape( (-1, np.prod(self.prev_shape))))

            dx = np.linalg.matmul(dy, self.weights.T)
            self.model.tracer.emit_event(PYDTNN_OPS_EVENT, PYDTNN_EVENT_FINISHED)            
            return dx
