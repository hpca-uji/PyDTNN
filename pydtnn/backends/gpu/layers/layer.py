from collections import abc

import numpy as np

from pydtnn.layers.layer import Layer
from pydtnn.tracers.events import PYDTNN_MDL_EVENT, PYDTNN_MDL_EVENTS, PYDTNN_OPS_EVENT, PYDTNN_OPS_EVENTS, \
    PYDTNN_EVENT_FINISHED, PYDTNN_MDL_EVENT_enum, PYDTNN_OPS_EVENT_enum
from pydtnn.utils.constants import ArrayShape

try:
    from pydtnn.libs.libmpi import MPI
except Exception as e:
    pass

try:
    import pydtnn.libs.libnccl as nccl
except Exception as e:
    pass

from numpy import ndarray
from pydtnn.backends.gpu.utils.tensor_gpu import TensorGPU

import pycuda.gpuarray as gpuarray  # type: ignore


class LayerGPU(Layer[TensorGPU]):
    """
    Extends a Layer class with the attributes and methods required by GPU Layers.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # GPU layer attributes
        # NOTE: All of these values will be initalized in the "initialize" method.
        self.weights_cpu: ndarray = None  # type: ignore
        self.biases_cpu: ndarray = None  # type: ignore
        self.dx: TensorGPU = None  # type: ignore
        self.dw: TensorGPU = None  # type: ignore
        self.db: TensorGPU = None  # type: ignore
        self.dw_cpu: ndarray = None  # type: ignore
        self.db_cpu: ndarray = None  # type: ignore
        self.one_vec_cpu: ndarray = None  # type: ignore
        self.one_vec_gpu: gpuarray.GPUArray = None  # type: ignore
        self.grid = None
        self.block = None
    
    def initialize(self, prev_shape: tuple[int, ...], x: TensorGPU | None = None) -> None:
        super().initialize(prev_shape, x)
        self.grid = self.model.cuda_grid
        self.block = self.model.cuda_block
    # ---

    @property
    def _ary_prop(self) -> set[str]:
        return {*self.grad_vars.keys(), *self.grad_vars.values()}

    def _export_prop(self, key: str):
        if key not in self._ary_prop:
            return super()._export_prop(key)

        gpu_ary = getattr(self, key).ary
        cpu_ary = np.asarray(gpu_ary.get(), dtype=np.float64, order="C", copy=True)
        return cpu_ary

    def _import_prop(self, key: str, value) -> None:
        if key not in self._ary_prop:
            return super()._import_prop(key, value)

        gpu_ary = getattr(self, key).ary
        cpu_ary = np.asarray(value.reshape(gpu_ary.shape), dtype=self.model.dtype, order="C", copy=None)
        gpu_ary.set(cpu_ary)

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

        for w_, dw_ in self.grad_vars.items():
            if self.model.enable_nccl:
                self.model.stream.synchronize()
                dw: TensorGPU = getattr(self, dw_)
                # TODO: decrypt
                setattr(self, dw_, dw)
            else:
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
