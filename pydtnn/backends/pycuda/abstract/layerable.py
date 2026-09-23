"""PyCUDA implementation of layerable components for distributed training."""

from pydtnn.abstract.layerable import Layerable
from pydtnn.backends.pycuda.abstract.base import BasePycuda
from pydtnn.utils.tensor_array import TensorArray
from pydtnn.tracers.events import (PYDTNN_EVENT_FINISHED, PYDTNN_OPS_EVENT,
                                   PYDTNN_OPS_EVENTS, OpsEventEnum)

try:
    import pydtnn.libs.nccl as nccl
except Exception:
    nccl = None

__all__ = ("LayerablePycuda",)


class LayerablePycuda(Layerable[TensorArray], BasePycuda):
    """Provides PyCUDA-specific weight reduction capabilities for distributed layers."""

    def _model_init(self, prev_shape: tuple[int, ...], x: TensorArray | None) -> None:
        super()._model_init(prev_shape, x)
        if self.model.use_nccl:
            self._state_reduce_async = self._state_reduce_async_nccl
            self._state_reduce_sync = self._state_reduce_sync_nccl
            self._state_reduce_wait = self._state_reduce_wait_nccl
        elif not self.model.use_mpi_cuda:
            self._state_reduce_async = self._state_reduce_async_cpu
            self._state_reduce_sync = self._state_reduce_sync_cpu
            self._state_reduce_wait = self._state_reduce_wait_cpu

    def _state_reduce_async_nccl(self, state_: str) -> None:
        state = getattr(self, state_)
        assert nccl is not None

        state.ary *= self.model.rank_weight
        # TODO: self.model._encode_reduce
        nccl.ncclAllReduce(
            state.ptr_voidp,
            state.ptr_voidp,
            state.size,
            self.model.nccl_type,
            nccl.RedOp.Sum,
            comm=self.model.nccl_comm,
            stream=self.model.stream.handle,
        )

        # NOTE: state where called "dw" before the renaming.
        # # Hierarchical mode NCCL + MPI
        # if len(self.model.inter_ranks) == 1:
        #     nccl.ncclAllReduce(dw.ptr, dw.ptr, dw.size, self.model.nccl_type,
        #                        nccl.RedOp.Sum, comm=self.model.nccl_comm,
        #                        stream=self.model.stream.handle)
        # else:
        #     # Hierarchical allreduce - Phase 1: ncclReduce + Iallreduce
        #     nccl.ncclReduce(dw.ptr, dw.ptr, dw.size, self.model.nccl_type,
        #                     nccl.RedOp.Sum, root=0, comm=self.model.nccl_comm,
        #                     stream=self.model.stream.handle)
        #     if self.model.rank in self.model.inter_ranks:
        #         self.model.stream.synchronize()
        #         req = self.model.inter_comm.Iallreduce(MPI.IN_PLACE, dw_cpu, op=MPI.SUM)

    def _state_reduce_wait_nccl(self, state_: str) -> None:
        # self.model.stream.synchronize()
        state: TensorArray = getattr(self, state_)
        # TODO: self.model._decode_reduce
        setattr(self, state_, state)

        # # Hierarchical mode NCCL + MPI
        # if self.model.use_nccl:
        #     if len(self.model.inter_ranks) == 1:
        #         # Do nothing, Allreduce was already completed in phase 1
        #         pass
        #     else:
        #         # Hierarchical allreduce - Phase 2: wait + ncclBroadcast
        #         if self.model.rank in self.model.inter_ranks:
        #             self.reqs_allred[dw_].wait()
        #         nccl.ncclBroadcast(dw.ptr, dw.ptr, dw.size, self.model.nccl_type,
        #                            root=0, comm=self.model.nccl_comm,
        #                            stream=self.model.stream.handle)

    def _state_reduce_sync_nccl(self, state_: str) -> None:
        # stream = self.model.stream.handle)
        state = getattr(self, state_)
        assert nccl is not None

        state.ary *= self.model.rank_weight
        # TODO: self.model._encode_reduce
        self.model.tracer.emit_event(
            PYDTNN_OPS_EVENT,
            self.id * PYDTNN_OPS_EVENTS + OpsEventEnum.OPS_ALLREDUCE_DW,
        )
        nccl.ncclAllReduce(
            state.ptr_voidp,
            state.ptr_voidp,
            state.size,
            self.model.nccl_type,
            nccl.RedOp.Sum,
            comm=self.model.nccl_comm,
            stream=self.model.stream.handle,
        )
        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, PYDTNN_EVENT_FINISHED)
        # TODO: self.mode._decode_reduce

        # # Hierarchical mode NCCL + MPI
        # if len(self.model.inter_ranks) == 1:
        #     # Only one node involved, perform ncclAllreduce across intra-node GPUs
        #     nccl.ncclAllReduce(dw.ptr, dw.ptr, dw.size, self.model.nccl_type,
        #                        nccl.RedOp.Sum, comm=self.model.nccl_comm,
        #                        stream=self.model.stream.handle)
        # else:
        #     # Hierarchical allreduce: ncclReduce + Allreduce + ncclBroadcast
        #     nccl.ncclReduce(dw.ptr, dw.ptr, dw.size, self.model.nccl_type,
        #                     nccl.RedOp.Sum, root=0, comm=self.model.nccl_comm,
        #                     stream=self.model.stream.handle)
        #     if self.model.rank in self.model.inter_ranks:
        #         if self.model.gpudirect:
        #             self.model.inter_comm.Allreduce(MPI.IN_PLACE, dw_cpu, op=MPI.SUM)
        #         else:
        #             dw_cpu = dw.get()
        #             self.model.inter_comm.Allreduce(MPI.IN_PLACE, dw_cpu, op=MPI.SUM)
        #             dw.set_async(dw_cpu, self.model.stream)
        #     nccl.ncclBroadcast(dw.ptr, dw.ptr, dw.size, self.model.nccl_type,
        #                        root=0, comm=self.model.nccl_comm,
        #                        stream=self.model.stream.handle)

    def _state_reduce_async(self, state_: str) -> None:
        state = getattr(self, state_)
        state_ary = f"{state_}_ary"
        setattr(self, state_ary, state.ary)
        super()._state_reduce_async(state_ary)

    def _state_reduce_wait(self, state_: str) -> None:
        state = getattr(self, state_)
        state_ary = f"{state_}_ary"
        super()._state_reduce_wait(state_ary)
        state_ary_ = getattr(self, state_ary)
        if state_ary_ is not state.ary:
            state.ary[:] = state_ary_
        delattr(self, state_ary)

    def _state_reduce_sync(self, state_: str) -> None:
        state = getattr(self, state_)
        state_ary = f"{state_}_ary"
        setattr(self, state_ary, state.ary)
        super()._state_reduce_sync(state_ary)
        state_ary_ = getattr(self, state_ary)
        if state_ary_ is not state.ary:
            state.ary[:] = state_ary_
        delattr(self, state_ary)

    def _state_reduce_async_cpu(self, state_: str) -> None:
        state = getattr(self, state_)
        state_cpu = f"{state_}_cpu"
        setattr(self, state_cpu, state.get())
        super()._state_reduce_async(state_cpu)

    def _state_reduce_wait_cpu(self, state_: str) -> None:
        state = getattr(self, state_)
        state_cpu = f"{state_}_cpu"
        super()._state_reduce_wait(state_cpu)
        state.set(getattr(self, state_cpu))
        delattr(self, state_cpu)

    def _state_reduce_sync_cpu(self, state_: str) -> None:
        state = getattr(self, state_)
        state_cpu = f"{state_}_cpu"
        setattr(self, state_cpu, state.get())
        super()._state_reduce_sync(state_cpu)
        state.set(getattr(self, state_cpu))
        delattr(self, state_cpu)
