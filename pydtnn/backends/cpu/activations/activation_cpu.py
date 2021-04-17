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

from pydtnn.activations.activation import Activation
from pydtnn.tracers import PYDTNN_MDL_EVENT, PYDTNN_MDL_EVENTS, PYDTNN_OPS_EVENT, PYDTNN_OPS_EVENTS, \
    PYDTNN_MDL_ALLREDUCE_DW, PYDTNN_OPS_ALLREDUCE_DW

try:
    # noinspection PyUnresolvedReferences
    from mpi4py import MPI
except (ImportError, ModuleNotFoundError):
    pass


class ActivationCPU(Activation, ABC):
    """
    Extends an Activation class with the attributes and methods required by CPU Activations.

    The next methods are copied from LayerCPU:
      * reduce_weights_async()
      * wait_allreduce_async()
      * reduce_weights_sync()
    """

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
