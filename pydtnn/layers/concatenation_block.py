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

from .layer import Layer
from ..performance_models import *
from ..tracers import PYDTNN_OPS_EVENT, PYDTNN_OPS_EVENTS, PYDTNN_MDL_FORWARD, PYDTNN_MDL_BACKWARD, PYDTNN_MDL_EVENT, \
    PYDTNN_MDL_EVENTS, PYDTNN_OPS_BACKWARD_ELTW_SUM, \
    PYDTNN_OPS_BACKWARD_SPLIT, PYDTNN_OPS_FORWARD_CONCAT, PYDTNN_OPS_FORWARD_REPLICATE


class ConcatenationBlock(Layer):

    def __init__(self, *args, **kwargs):
        super().__init__(**kwargs)
        self.paths = []
        for p in args:
            self.paths.append(p)
        self.is_block_layer = True
        self.out_shapes = []
        # The next attributes will be initialized later
        self.out_co = None
        self.idx_co = None

    def initialize(self, prev_shape, need_dx=True):
        super().initialize(prev_shape, need_dx)
        for p in self.paths:
            for i, layer in enumerate(p):
                layer.set_model(self.model)
                layer.initialize(prev_shape, need_dx)
                prev_shape = layer.shape
                self.fwd_time += layer.fwd_time
                self.bwd_time += layer.bwd_time
                self.nparams += layer.nparams
            self.out_shapes.append(prev_shape)
            prev_shape = self.prev_shape
        assert all([tuple(o[1:]) == tuple(self.out_shapes[0][1:]) for o in self.out_shapes])
        self.out_co = [s[0] for s in self.out_shapes]
        self.idx_co = np.cumsum(self.out_co, axis=0)
        self.shape = (sum(self.out_co), *self.out_shapes[0][1:])

    def show(self, attrs=""):
        print(
            f"|{self.id:^7d}"
            f"|{(type(self).__name__.replace('Concatenation', 'Concat') + ' (%d-path)' % len(self.paths)):^26s}"
            f"|{'':9s}|{str(self.shape):^15s}|{'':19s}|{'':24s}|")
        for i, p in enumerate(self.paths):
            print(f"|{('Path %d' % i):^7s}|{'':^26s}|{'':9s}|{'':15s}|{'':19s}|{'':24s}|")
            for layer in p:
                layer.show()

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

    def forward(self, x):
        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, self.id * PYDTNN_OPS_EVENTS + PYDTNN_OPS_FORWARD_REPLICATE)
        x = [x] * len(self.paths)
        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, 0)

        for i, p in enumerate(self.paths):
            for layer in p:
                self.model.tracer.emit_event(PYDTNN_MDL_EVENT, layer.id * PYDTNN_MDL_EVENTS + PYDTNN_MDL_FORWARD)
                x[i] = layer.forward(x[i])
                self.model.tracer.emit_event(PYDTNN_MDL_EVENT, 0)

        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, self.id * PYDTNN_OPS_EVENTS + PYDTNN_OPS_FORWARD_CONCAT)
        y = np.concatenate(x, axis=1)
        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, 0)

        return y

    def backward(self, dy):
        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, self.id * PYDTNN_OPS_EVENTS + PYDTNN_OPS_BACKWARD_SPLIT)
        dx = np.split(dy, self.idx_co[:-1], axis=1)
        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, 0)

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
