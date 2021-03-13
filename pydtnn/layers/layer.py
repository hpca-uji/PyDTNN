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
from abc import ABC

from .. import activations
from .. import model
from ..performance_models import *
from ..tracers import PYDTNN_MDL_EVENT, PYDTNN_MDL_EVENTS, PYDTNN_OPS_EVENT, PYDTNN_OPS_EVENTS, \
    PYDTNN_MDL_ALLREDUCE_DW, PYDTNN_OPS_ALLREDUCE_DW

try:
    # noinspection PyUnresolvedReferences
    from mpi4py import MPI
    # import pycuda.autoinit
    import pycuda.gpuarray as gpuarray
    import pycuda.driver as drv
    import skcuda.linalg as culinalg
    import skcuda.misc as cumisc
    # noinspection PyUnresolvedReferences
    import libnccl.libnccl as nccl
except ModuleNotFoundError:
    pass
except ImportError:
    pass


class Layer(ABC):

    def __new__(cls, *args, **kwargs):
        # If GPU is requested, return a GPU-related object instead
        if not model.enable_cudnn:
            new_cls = cls
        else:
            module_name = "activations" if cls in activations.__dict__.values() else "layers"
            module = importlib.import_module(f"{module_name}_gpu")
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
        self.x = None
        self.reqs_allred = {}
        self.stream_2 = None
        self.db = None
        self.dw = None

    def initialize(self, prev_shape, need_dx=True, x=None):
        self.prev_shape = prev_shape
        self.need_dx = need_dx
        self.x = x

    def set_model(self, parent_model, layer_id):
        self.model = parent_model
        self.id = layer_id

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

        # if self.model.enable_cudnn:
        #     if self.model.enable_nccl or self.model.gpudirect:
        #        self.model.stream.synchronize()
        #     else:
        #        self.stream_2.synchronize()

        for w_, dw_ in self.grad_vars.items():
            dw = getattr(self, dw_)

            if self.model.enable_cudnn:
                if self.model.enable_nccl:
                    self.model.stream.synchronize()
                    nccl.ncclAllReduce(dw.ptr, dw.ptr, dw.size, self.model.nccl_type,
                                       nccl.RedOp.Sum, comm=self.model.nccl_comm,
                                       stream=self.stream_2.handle)
                    req = None

                    # # Hierarchical mode NCCL + MPI
                    # if len(self.model.inter_ranks) == 1:
                    #     nccl.ncclAllReduce(dw.ptr, dw.ptr, dw.size, self.model.nccl_type, 
                    #                        nccl.RedOp.Sum, comm=self.model.nccl_comm, 
                    #                        stream=self.stream_2.handle)
                    #
                    # else:
                    #     # Hierarchical allreduce - Phase 1: ncclReduce + Iallreduce
                    #     nccl.ncclReduce(dw.ptr, dw.ptr, dw.size, self.model.nccl_type, 
                    #                     nccl.RedOp.Sum, root=0, comm=self.model.nccl_comm, 
                    #                     stream=self.stream_2.handle)
                    #
                    #     if self.model.rank in self.model.inter_ranks:
                    #         if not self.model.gpudirect:
                    #             dw.ary.get_async(self.stream_2, dw_cpu)
                    #
                    #         self.stream_2.synchronize()
                    #         req = self.model.inter_comm.Iallreduce(MPI.IN_PLACE, dw_cpu, op=MPI.SUM) 

                else:  # Without NCCL

                    # We have asynchronously moved the dw and db to dw_cpu and db_cpu in stream_2
                    # so we need to synchronize stream_2 before performing Allreduce.
                    # In GPU direct we have to synchronize the main stream to ensure dw and db are ready.

                    if not self.model.gpudirect:
                        self.stream_2.synchronize()
                    else:
                        self.model.stream.synchronize()

                    dw_cpu = getattr(self, "%s_cpu" % dw_)
                    req = self.model.comm.Iallreduce(MPI.IN_PLACE, dw_cpu, op=MPI.SUM)

            else:  # Without GPUs, only MPI
                req = self.model.comm.Iallreduce(MPI.IN_PLACE, dw, op=MPI.SUM)

            self.reqs_allred[dw_] = req

    def wait_allreduce_async(self):
        if not self.model.comm or self.model.enable_nccl:
            return

        for w_, dw_ in self.grad_vars.items():
            self.reqs_allred[dw_].wait()

            # # Hierarchical mode NCCL + MPI
            # if self.model.enable_nccl:  
            #     if len(self.model.inter_ranks) == 1: 
            #         # Do nothing, Allreduce was already completed in phase 1
            #         pass
            #     else:
            #         # Hierarchical allreduce - Phase 2: wait + ncclBroadcast
            #         if self.model.rank in self.model.inter_ranks:
            #             self.reqs_allred[dw_].wait()
            #             if not self.model.gpudirect: 
            #                 dw.ary.set_async(dw_cpu, self.stream_2)
            #     
            #         nccl.ncclBroadcast(dw.ptr, dw.ptr, dw.size, self.model.nccl_type, 
            #                            root=0, comm=self.model.nccl_comm, 
            #                            stream=self.stream_2.handle)

            if self.model.enable_cudnn and not self.model.gpudirect:
                dw = getattr(self, dw_)
                dw_cpu = getattr(self, f"{dw_}_cpu")

                # If there is no CUDA-aware MPI, copy data back to GPU
                dw.ary.set_async(dw_cpu, self.stream_2)

    def reduce_weights_sync(self):
        if not self.model.comm:
            return

        for w_, dw_ in self.grad_vars.items():
            self.model.tracer.emit_nevent([PYDTNN_MDL_EVENT, PYDTNN_OPS_EVENT],
                                          [self.id * PYDTNN_MDL_EVENTS + PYDTNN_MDL_ALLREDUCE_DW,
                                           self.id * PYDTNN_OPS_EVENTS + PYDTNN_OPS_ALLREDUCE_DW])
            dw = getattr(self, dw_)

            if self.model.enable_cudnn:

                if self.model.enable_nccl:
                    nccl.ncclAllReduce(dw.ptr, dw.ptr, dw.size, self.model.nccl_type,
                                       nccl.RedOp.Sum, comm=self.model.nccl_comm,
                                       stream=self.stream_2.handle)

                    # # Hierarchical mode NCCL + MPI
                    # if len(self.model.inter_ranks) == 1:
                    #     # Only one node involved, perform ncclAllreduce across intra-node GPUs
                    #     nccl.ncclAllReduce(dw.ptr, dw.ptr, dw.size, self.model.nccl_type, 
                    #                        nccl.RedOp.Sum, comm=self.model.nccl_comm, 
                    #                        stream=self.stream_2.handle)
                    # else:
                    #     # Hierarchical allreduce: ncclReduce + Allreduce + ncclBroadcast
                    #     nccl.ncclReduce(dw.ptr, dw.ptr, dw.size, self.model.nccl_type, 
                    #                     nccl.RedOp.Sum, root=0, comm=self.model.nccl_comm,
                    #                     stream=self.stream_2.handle)
                    # 
                    #     self.stream_2.synchronize()
                    #     if self.model.rank in self.model.inter_ranks:
                    #         if self.model.gpudirect: 
                    #             self.model.inter_comm.Allreduce(MPI.IN_PLACE, dw_cpu, op=MPI.SUM) 
                    #         else:
                    #             dw_cpu = dw.ary.get()
                    #             self.model.inter_comm.Allreduce(MPI.IN_PLACE, dw_cpu, op=MPI.SUM)
                    #             dw.ary.set_async(dw_cpu, self.stream_2)
                    # 
                    #     nccl.ncclBroadcast(dw.ptr, dw.ptr, dw.size, self.model.nccl_type, 
                    #                        root=0, comm=self.model.nccl_comm, 
                    #                        stream=self.stream_2.handle)

                else:  # Without NCCL

                    # We have asynchronously moved the dw and db to dw_cpu and db_cpu in stream_2
                    # so we need to synchronize stream_2 before performing Allreduce.
                    # In GPU direct, the main stream is already synchronized.

                    if not self.model.gpudirect:
                        self.stream_2.synchronize()

                    dw_cpu = getattr(self, f"{dw_}_cpu")
                    self.model.comm.Allreduce(MPI.IN_PLACE, dw_cpu, op=MPI.SUM)

                    if not self.model.gpudirect:
                        dw.ary.set_async(dw_cpu, self.stream_2)
            else:
                self.model.comm.Allreduce(MPI.IN_PLACE, dw, op=MPI.SUM)

            self.model.tracer.emit_nevent([PYDTNN_MDL_EVENT, PYDTNN_OPS_EVENT], [0, 0])
