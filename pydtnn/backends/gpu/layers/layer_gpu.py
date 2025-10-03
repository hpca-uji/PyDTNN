from abc import ABC
from collections import abc

from pydtnn.layers.layer import Layer
from pydtnn.tracers import  PYDTNN_MDL_EVENT, PYDTNN_MDL_EVENTS, PYDTNN_OPS_EVENT, PYDTNN_OPS_EVENTS, \
                            PYDTNN_EVENT_FINISHED, PYDTNN_MDL_EVENT_enum, PYDTNN_OPS_EVENT_enum

try:
    # noinspection PyUnresolvedReferences
    from pydtnn.comm import MPI
except (ImportError, ModuleNotFoundError):
    pass

try:
    # noinspection PyUnresolvedReferences
    import pydtnn.backends.gpu.libs.libnccl as nccl
except (ImportError, ModuleNotFoundError, OSError):
    pass

from numpy import ndarray
from ..tensor_gpu import TensorGPU

class LayerGPU(Layer, ABC):
    """
    Extends a Layer class with the attributes and methods required by GPU Layers.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # GPU layer attributes
        self.y: TensorGPU = None
        self.weights_cpu: ndarray = None
        self.biases_cpu: ndarray = None
        self.x: TensorGPU = None
        self.dx: TensorGPU = None
        self.dw: TensorGPU = None
        self.db: TensorGPU = None
        self.dw_cpu: ndarray = None
        self.db_cpu: ndarray = None
        self.one_vec_cpu: ndarray = None
        self.one_vec_gpu: TensorGPU = None

    # noinspection PyMethodOverriding
    def initialize(self, prev_shape: tuple[int, ...], x: TensorGPU) -> None:
        self.x = x  # Must be before super().initialize()
        super().initialize(prev_shape)

    def reduce_weights_async(self, gradient=True):
        if not self.model.comm:
            return
        self.reqs_allred = {}

        # if self.model.enable_cudnn:
        #     if self.model.enable_nccl or self.model.gpudirect:
        #        self.model.stream.synchronize()
        #     else:
        #        self.stream_2.synchronize()

        for w_, dw_ in self.grad_vars.items():
            dw_ = dw_ if gradient else w_
            dw = getattr(self, dw_)

            if self.model.enable_nccl:
                self.model.stream.synchronize()
                dw *= self.model.rank_weight
                # TODO: crypt
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
                dw_cpu *= self.model.rank_weight
                if self.model.crypt:
                    dw_cpu = self.model.crypt.encrypt(dw_cpu)
                if isinstance(dw_cpu, abc.Buffer):
                    req = self.model.comm.Iallreduce(MPI.IN_PLACE, dw_cpu, op=MPI.SUM)
                else:
                    req = self.model.comm.iallreduce(dw_cpu, op=MPI.SUM)
                self.reqs_allred[dw_] = req

    def wait_allreduce_async(self, gradient=True):
        if not self.model.comm:
            return

        if self.model.enable_nccl:
            self.model.stream.synchronize()
            dw = getattr(self, dw_)
            # TODO: decrypt
            setattr(self, dw_, dw)
        else:
            for w_, dw_ in self.grad_vars.items():
                dw_ = dw_ if gradient else w_
                self.reqs_allred[dw_].wait()
                dw = getattr(self, dw_)
                res = self.reqs_allred[dw_].wait()
                if res is None:
                    dw = getattr(self, dw_)
                else:
                    dw = res
                if self.model.crypt:
                    dw = self.model.crypt.decrypt(dw)
                setattr(self, dw_, dw)

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

    def reduce_weights_sync(self, gradient=True):
        if not self.model.comm:
            return

        for w_, dw_ in self.grad_vars.items():
            dw_ = dw_ if gradient else w_
            self.model.tracer.emit_nevent([PYDTNN_MDL_EVENT, PYDTNN_OPS_EVENT],
                                          [self.id * PYDTNN_MDL_EVENTS + PYDTNN_MDL_EVENT_enum.ALLREDUCE_DW,
                                           self.id * PYDTNN_OPS_EVENTS + PYDTNN_OPS_EVENT_enum.OPS_ALLREDUCE_DW])
            # stream = self.stream_2.handle)
            dw = getattr(self, dw_)

            if self.model.enable_nccl:
                dw *= self.model.rank_weight
                self.stream_2.synchronize()
                # TODO: crypt
                nccl.ncclAllReduce(dw.ptr, dw.ptr, dw.size, self.model.nccl_type,
                                   nccl.RedOp.Sum, comm=self.model.nccl_comm,
                                   stream=self.stream_2.handle)
                self.stream_2.synchronize()
                # TODO: decrypt

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
                dw_cpu *= self.model.rank_weight
                if self.model.crypt:
                    dw_cpu = self.model.crypt.encrypt(dw_cpu)
                if self.model.use_mpi_buffers:
                    self.model.comm.Allreduce(MPI.IN_PLACE, dw_cpu, op=MPI.SUM)
                else:
                    dw_cpu = self.model.comm.allreduce(dw_cpu, op=MPI.SUM)
                if self.model.crypt:
                    dw_cpu = self.model.crypt.decrypt(dw_cpu)
                setattr(self, f"{dw_}_cpu", dw_cpu)

                if not self.model.gpudirect:
                    dw.ary.set_async(dw_cpu, self.stream_2)

            self.model.tracer.emit_nevent([PYDTNN_MDL_EVENT, PYDTNN_OPS_EVENT], [PYDTNN_EVENT_FINISHED, PYDTNN_EVENT_FINISHED])
