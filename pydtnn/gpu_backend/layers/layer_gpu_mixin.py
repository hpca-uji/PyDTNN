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

from pydtnn.tracers import PYDTNN_MDL_EVENT, PYDTNN_MDL_EVENTS, PYDTNN_OPS_EVENT, PYDTNN_OPS_EVENTS, \
    PYDTNN_MDL_ALLREDUCE_DW, PYDTNN_OPS_ALLREDUCE_DW
import pydtnn.gpu_backend.libs.libnccl as nccl

try:
    # noinspection PyUnresolvedReferences
    from mpi4py import MPI
    # noinspection PyUnresolvedReferences
    import libnccl.libnccl as nccl
except (ImportError, ModuleNotFoundError, OSError):
    pass

class LayerGPUMixin:
    """
    Mixin used to extend a Layer class with the attributes and methods required
    by GPU Layers.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.x = None
        self.y = None
        self.weights_cpu = None
        self.biases_cpu = None
        self.x = None
        self.dx = None
        self.dw = None
        self.db = None
        self.dw_cpu = None
        self.db_cpu = None
        self.one_vec_cpu = None
        self.one_vec_gpu = None

    def initialize(self, prev_shape, need_dx, x):
        super().initialize(prev_shape, need_dx)
        self.need_dx = need_dx
        self.x = x


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

            if self.model.enable_nccl:
                self.model.stream.synchronize()
                nccl.ncclAllReduce(dw.ptr, dw.ptr, dw.size, self.model.nccl_type,
                                   nccl.RedOp.Sum, comm=self.model.nccl_comm,
                                   stream=self.stream_2.handle)

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

                dw_cpu = getattr(self, f"{dw_}_cpu")
                req = self.model.comm.Iallreduce(MPI.IN_PLACE, dw_cpu, op=MPI.SUM)
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

            if not self.model.gpudirect:
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
                                           #stream = self.stream_2.handle)
            dw = getattr(self, dw_)

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

            self.model.tracer.emit_nevent([PYDTNN_MDL_EVENT, PYDTNN_OPS_EVENT], [0, 0])
