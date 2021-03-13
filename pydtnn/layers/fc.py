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
from .. import activations
from .. import initializers
from ..model import TRAIN_MODE
from ..performance_models import *
from ..tracers import PYDTNN_OPS_EVENT, PYDTNN_OPS_EVENTS, PYDTNN_OPS_COMP_DW_MATMUL, PYDTNN_OPS_COMP_DX_MATMUL, \
    PYDTNN_OPS_FORWARD_MATMUL


class FC(Layer):

    def __init__(self, shape=(1,), activation="", use_bias=True,
                 weights_initializer="glorot_uniform",
                 biases_initializer="zeros"):
        super(FC, self).__init__(shape)
        self.act = getattr(activations, activation, None)
        self.use_bias = use_bias
        self.weights_initializer = getattr(initializers, weights_initializer)
        self.biases_initializer = getattr(initializers, biases_initializer)
        self.grad_vars = {"weights": "dw"}
        if self.use_bias:
            self.grad_vars["biases"] = "db"
        # The next attributes will be initialized later
        self.x = None

    def initialize(self, prev_shape, need_dx=True):
        super().initialize(prev_shape, need_dx)
        self.weights = self.weights_initializer((*prev_shape, *self.shape), self.model.dtype)
        if self.use_bias:
            self.biases = self.biases_initializer(self.shape, self.model.dtype)
        self.nparams = self.weights.size + (self.biases.size if self.use_bias else 0)

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

    def show(self, attrs=""):
        super().show("|{:^19s}|{:^24s}|".format(str(self.weights.shape), ""))

    def forward(self, x):
        if self.model.mode == TRAIN_MODE:
            self.x = x
        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, self.id * PYDTNN_OPS_EVENTS + PYDTNN_OPS_FORWARD_MATMUL)
        res = self.model.matmul(x, self.weights)
        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, 0)
        return res + self.biases if self.use_bias else 0

    def backward(self, dy):
        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, self.id * PYDTNN_OPS_EVENTS + PYDTNN_OPS_COMP_DW_MATMUL)
        self.dw = self.model.matmul(self.x.T, dy)
        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, 0)

        if self.use_bias:
            self.db = np.sum(dy, axis=0)

        if self.need_dx:
            self.model.tracer.emit_event(PYDTNN_OPS_EVENT, self.id * PYDTNN_OPS_EVENTS + PYDTNN_OPS_COMP_DX_MATMUL)
            dx = self.model.matmul(dy, self.weights.T)
            self.model.tracer.emit_event(PYDTNN_OPS_EVENT, 0)
            return dx
