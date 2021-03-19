"""
PyDTNN Layer base class
"""

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

import importlib
import inspect
from abc import ABC, abstractmethod

import numpy as np

from .. import model as model_module
from ..tracers import PYDTNN_MDL_EVENT, PYDTNN_MDL_EVENTS, PYDTNN_OPS_EVENT, PYDTNN_OPS_EVENTS, \
    PYDTNN_MDL_ALLREDUCE_DW, PYDTNN_OPS_ALLREDUCE_DW

try:
    # noinspection PyUnresolvedReferences
    from mpi4py import MPI
    # noinspection PyUnresolvedReferences
    import libnccl.libnccl as nccl
except (ImportError, ModuleNotFoundError, OSError):
    pass


class Layer(ABC):

    def __new__(cls, *args, **kwargs):
        if not model_module.enable_cudnn:
            new_cls = cls
        else:
            # If GPU is requested, return a GPU-related object instead
            ancestors_names = [x.__name__ for x in inspect.getmro(cls)]
            module_name = "activations" if "Activation" in ancestors_names else "layers"
            module = importlib.import_module(f"..gpu_backend.{module_name}", package="pydtnn.layers")
            new_cls = getattr(module, f"{cls.__name__}GPU")
        instance = super(Layer, new_cls).__new__(new_cls)
        if new_cls != cls:
            instance.__init__(*args, **kwargs)
        return instance

    def __init__(self, shape=()):
        self.nparams = 0
        self.shape = shape
        self.weights = np.array([])
        self.biases = np.array([])
        self.act = None
        self.grad_vars = {}
        self.fwd_time = np.zeros((4,), dtype=np.float32)
        self.bwd_time = np.zeros((4,), dtype=np.float32)
        self.paths = []
        # The next attributes will be initialized later
        self.id = None
        self.model = None
        self.prev_shape = None
        self.need_dx = True
        self.is_block_layer = False
        self.reqs_allred = {}
        self.stream_2 = None

    def set_model(self, parent_model):
        self.model = parent_model
        self.id = next(self.model.layer_id)

    def initialize(self, prev_shape, need_dx=True):
        self.prev_shape = prev_shape
        self.need_dx = need_dx

    @abstractmethod
    def forward(self, x):
        pass

    @abstractmethod
    def backward(self, dy):
        pass

    def show(self, attrs=""):
        if not attrs:
            attrs = "|{:19s}|{:^24s}|".format("", "")
        print(f"|{self.id:^7d}|{type(self).__name__:^26s}|{self.nparams:^9d}|{str(self.shape):^15}" + attrs)

    @property
    def children(self):
        children = []
        for path in self.paths:
            children += [layer for layer in path]
        return children

    def update_weights(self, optimizer):
        optimizer.update(self)

    def reduce_weights_async(self):
        if not self.model.comm:
            return
        self.reqs_allred = {}

        for w_, dw_ in self.grad_vars.items():
            dw = getattr(self, dw_)
            req = self.model.comm.Iallreduce(MPI.IN_PLACE, dw, op=MPI.SUM)
            self.reqs_allred[dw_] = req

    def wait_allreduce_async(self):
        if not self.model.comm or self.model.enable_nccl:
            return
        for w_, dw_ in self.grad_vars.items():
            self.reqs_allred[dw_].wait()

    def reduce_weights_sync(self):
        if not self.model.comm:
            return
        for w_, dw_ in self.grad_vars.items():
            self.model.tracer.emit_nevent([PYDTNN_MDL_EVENT, PYDTNN_OPS_EVENT],
                                          [self.id * PYDTNN_MDL_EVENTS + PYDTNN_MDL_ALLREDUCE_DW,
                                           self.id * PYDTNN_OPS_EVENTS + PYDTNN_OPS_ALLREDUCE_DW])
            dw = getattr(self, dw_)
            self.model.comm.Allreduce(MPI.IN_PLACE, dw, op=MPI.SUM)
            self.model.tracer.emit_nevent([PYDTNN_MDL_EVENT, PYDTNN_OPS_EVENT], [0, 0])
