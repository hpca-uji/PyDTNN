"""PyCUDA implementation of layerable components for distributed training."""

from pydtnn.abstract.layerable import Layerable
from pydtnn.backends.pycuda.abstract.base import BasePycuda
from pydtnn.backends.pycuda.utils.tensor_array import TensorArray
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
            self._reduce_weights_async = self._reduce_weights_async_nccl
            self._reduce_weights_sync = self._reduce_weights_sync_nccl
            self._wait_allreduce_async = self._wait_allreduce_async_nccl
        else:
            self._reduce_weights_async = self._reduce_weights_async_no_nccl
            self._reduce_weights_sync = self._reduce_weights_sync_no_nccl
            self._wait_allreduce_async = self._wait_allreduce_async_no_nccl

    def reduce_weights_async(self, gradient: bool = True) -> None:
        """Initiates asynchronous weight reduction across distributed processes."""
        # NOTE: Keep in sync with Layer
        if not self.model.comm:
            return

        # if self.model.use_cudnn:
        #     if self.model.use_nccl or self.model.gpudirect:
        #        self.model.stream.synchronize()
        #     else:
        #        self.stream_2.synchronize()

        # NOTE:  self.grad_vars = {[VAR]: [VAR's GRADIENT]}
        vars_to_iterate = self.grad_vars.values() if gradient else self.grad_vars.keys()

        for w_dw in vars_to_iterate:
            self._reduce_weights_async(w_dw)

    def _reduce_weights_async_nccl(self, weights_: str) -> None:
        dw = getattr(self, weights_)
        assert nccl is not None

        # self.model.stream.synchronize()
        dw *= self.model.rank_weight
        # TODO: self.model._encode_reduce
        nccl.ncclAllReduce(
            dw.ptr,
            dw.ptr,
            dw.size,
            self.model.nccl_type,
            nccl.RedOp.Sum,
            comm=self.model.nccl_comm,
            stream=self.stream_2.handle,
        )

        # # Hierarchical mode NCCL + MPI
        # if len(self.model.inter_ranks) == 1:
        #     nccl.ncclAllReduce(dw.ptr, dw.ptr, dw.size, self.model.nccl_type,
        #                        nccl.RedOp.Sum, comm=self.model.nccl_comm,
        #                        stream=self.stream_2.handle)

        # else:
        #     # Hierarchical allreduce - Phase 1: ncclReduce + Iallreduce
        #     nccl.ncclReduce(dw.ptr, dw.ptr, dw.size, self.model.nccl_type,
        #                     nccl.RedOp.Sum, root=0, comm=self.model.nccl_comm,
        #                     stream=self.stream_2.handle)

        #     if self.model.rank in self.model.inter_ranks:
        #         if not self.model.gpudirect:
        #             dw.get_async(self.stream_2, dw_cpu)

        #         self.stream_2.synchronize()
        #         req = self.model.inter_comm.Iallreduce(MPI.IN_PLACE, dw_cpu, op=MPI.SUM)

    def _reduce_weights_async_no_nccl(self, weights_: str) -> None:
        # Without NCCL
        # We have asynchronously moved the dw and db to dw_cpu and db_cpu in stream_2
        # so we need to synchronize stream_2 before performing Allreduce.
        # In GPU direct we have to synchronize the main stream to ensure dw and db
        # are ready.

        if not self.model.use_gpudirect:
            self.stream_2.synchronize()

        dw_cpu = getattr(self, f"{weights_}_cpu")
        # NOTE: Desde aquí es igual a la versión de Numpy
        self.model.tracer.emit_event(
            PYDTNN_OPS_EVENT,
            self.id * PYDTNN_OPS_EVENTS + OpsEventEnum.LAYER_ENCODE,
        )
        dw_cpu = self.model._layer_reduce_encode(dw_cpu)
        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, PYDTNN_EVENT_FINISHED)

        assert weights_ not in self.reqs_allred, f"MPI request overwritten ({weights_} not waited)!"
        req = self.model._layer_reduce_async(dw_cpu)
        self.reqs_allred[weights_] = req
        # NOTE: Has aquí es igual a la versión de Numpy

    def wait_allreduce_async(self, gradient: bool = True) -> None:
        """Waits for completion of asynchronous weight reduction operations."""
        # NOTE: Keep in sync with Layer
        if not self.model.comm:
            return

        # NOTE:  self.grad_vars = {[VAR]: [VAR's GRADIENT]}
        vars_to_iterate = self.grad_vars.values() if gradient else self.grad_vars.keys()

        for w_dw in vars_to_iterate:
            self._wait_allreduce_async(w_dw)

    def _wait_allreduce_async_nccl(self, weights_: str) -> None:
        # self.model.stream.synchronize()
        weights: TensorArray = getattr(self, weights_)
        # TODO: self.model._decode_reduce
        setattr(self, weights_, weights)
        # # Hierarchical mode NCCL + MPI
        # if self.model.use_nccl:
        #     if len(self.model.inter_ranks) == 1:
        #         # Do nothing, Allreduce was already completed in phase 1
        #         pass
        #     else:
        #         # Hierarchical allreduce - Phase 2: wait + ncclBroadcast
        #         if self.model.rank in self.model.inter_ranks:
        #             self.reqs_allred[dw_].wait()
        #             if not self.model.gpudirect:
        #                 dw.set_async(dw_cpu, self.stream_2)

        #         nccl.ncclBroadcast(dw.ptr, dw.ptr, dw.size, self.model.nccl_type,
        #                            root=0, comm=self.model.nccl_comm,
        #                            stream=self.stream_2.handle)

    def _wait_allreduce_async_no_nccl(self, weights_: str) -> None:
        weights_cpu = getattr(self, f"{weights_}_cpu")
        req = self.reqs_allred.pop(weights_, None)
        if req is None:
            return
        weights_cpu = self.model._layer_reduce_wait(weights_cpu, req)

        self.model.tracer.emit_event(
            PYDTNN_OPS_EVENT,
            self.id * PYDTNN_OPS_EVENTS + OpsEventEnum.LAYER_DECODE,
        )
        weights_cpu = self.model._layer_reduce_decode(
            weights_cpu
        )  # FIXME: dw and dw_cpu relation unclear
        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, PYDTNN_EVENT_FINISHED)
        setattr(self, f"{weights_}_cpu", weights_cpu)

        dw = getattr(self, weights_)
        weights_cpu = getattr(self, f"{weights_}_cpu")

        # If there is no CUDA-aware MPI, copy data back to GPU
        dw.set_async(weights_cpu, self.stream_2)

    def reduce_weights_sync(self, gradient: bool = True) -> None:
        """Performs synchronous weight reduction across distributed processes."""
        # NOTE: Keep in sync with Layer
        if not self.model.comm:
            return

        # NOTE:  self.grad_vars = {[VAR]: [VAR's GRADIENT]}
        vars_to_iterate = self.grad_vars.values() if gradient else self.grad_vars.keys()

        for w_dw in vars_to_iterate:
            self._reduce_weights_sync(w_dw)

    def _reduce_weights_sync_nccl(self, weights_: str) -> None:
        # stream = self.stream_2.handle)
        weights = getattr(self, weights_)

        assert nccl is not None

        # self.stream_2.synchronize()
        weights *= self.model.rank_weight
        # TODO: self.model._encode_reduce
        self.model.tracer.emit_event(
            PYDTNN_OPS_EVENT,
            self.id * PYDTNN_OPS_EVENTS + OpsEventEnum.OPS_ALLREDUCE_DW,
        )
        nccl.ncclAllReduce(
            weights.ptr,
            weights.ptr,
            weights.size,
            self.model.nccl_type,
            nccl.RedOp.Sum,
            comm=self.model.nccl_comm,
            stream=self.stream_2.handle,
        )
        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, PYDTNN_EVENT_FINISHED)
        # self.stream_2.synchronize()
        # TODO: self.mode._decode_reduce

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

        #     self.stream_2.synchronize()
        #     if self.model.rank in self.model.inter_ranks:
        #         if self.model.gpudirect:
        #             self.model.inter_comm.Allreduce(MPI.IN_PLACE, dw_cpu, op=MPI.SUM)
        #         else:
        #             dw_cpu = dw.get()
        #             self.model.inter_comm.Allreduce(MPI.IN_PLACE, dw_cpu, op=MPI.SUM)
        #             dw.set_async(dw_cpu, self.stream_2)

        #     nccl.ncclBroadcast(dw.ptr, dw.ptr, dw.size, self.model.nccl_type,
        #                        root=0, comm=self.model.nccl_comm,
        #                        stream=self.stream_2.handle)

    def _reduce_weights_sync_no_nccl(self, weights_: str) -> None:
        # stream = self.stream_2.handle)
        weights = getattr(self, weights_)

        if not self.model.use_gpudirect:
            self.stream_2.synchronize()

        # NOTE: Desde aquí, igual que el de numpy, pero trabajando con "{weights_}_cpu" en vez de con "{weights_}"
        weights_cpu = getattr(self, f"{weights_}_cpu")

        self.model.tracer.emit_event(
            PYDTNN_OPS_EVENT,
            self.id * PYDTNN_OPS_EVENTS + OpsEventEnum.LAYER_ENCODE,
        )
        weights_cpu = self.model._layer_reduce_encode(weights_cpu)
        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, PYDTNN_EVENT_FINISHED)

        self.model.tracer.emit_event(
            PYDTNN_OPS_EVENT,
            self.id * PYDTNN_OPS_EVENTS + OpsEventEnum.OPS_ALLREDUCE_DW,
        )
        weights_cpu = self.model._layer_reduce_sync(weights_cpu)
        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, PYDTNN_EVENT_FINISHED)

        self.model.tracer.emit_event(
            PYDTNN_OPS_EVENT,
            self.id * PYDTNN_OPS_EVENTS + OpsEventEnum.LAYER_DECODE,
        )
        weights_cpu = self.model._layer_reduce_decode(weights_cpu)
        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, PYDTNN_EVENT_FINISHED)

        setattr(self, f"{weights_}_cpu", weights_cpu)
        # NOTE: Hasta aquí, igual que el de numpy, pero trabajando con "{weights_}_cpu" en vez de con "{weights_}"

        # If there is no CUDA-aware MPI, copy data back to GPU
        weights.set_async(weights_cpu, self.stream_2)
