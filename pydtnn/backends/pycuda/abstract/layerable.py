"""PyCUDA implementation of layerable components for distributed training."""

import ctypes
import numpy as np
import pycuda.driver as drv  # pyright: ignore[reportAttributeAccessIssue]
from pycuda import gpuarray  # pyright: ignore[reportAttributeAccessIssue]
from pydtnn.abstract.layerable import Layerable
from pydtnn.backends.pycuda.abstract.base import BasePycuda
from pydtnn.tracers.events import (PYDTNN_EVENT_FINISHED, PYDTNN_OPS_EVENT,
                                   PYDTNN_OPS_EVENTS, OpsEventEnum)
from pydtnn.utils.constants import ArrayShape
from pydtnn.utils.tensor_array import TensorArray

try:
    from pydtnn.libs import nccl
except Exception:
    nccl = None

__all__ = ("LayerablePycuda",)


class LayerablePycuda(Layerable[TensorArray], BasePycuda):
    """Provides PyCUDA-specific weight reduction capabilities for distributed layers."""

    def __init__(self, shape: ArrayShape = ()) -> None:
        """Init method of LayerablePycuda."""
        super().__init__(shape)
        # NOTE: It will be initialized later.
        self.y_model = None  # pyright: ignore[reportAttributeAccessIssue]

    def _model_init(self, prev_shape: ArrayShape, x: TensorArray | None) -> None:
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
            self.id * PYDTNN_OPS_EVENTS + OpsEventEnum.ALLREDUCE,
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

    def _post_init(self) -> None:
        super()._post_init()

        if self.id != self.model.backend.id:
            return None

        # NOTE: The actual "self.model.output_shape" is only available after initializating all layers.
        tensor_ary = TensorArray(
            gpuarray.empty((self.model.batch_size, *self.model.output_shape), self.model.dtype),
            self.model.tensor_format,
            self.model.cudnn_dtype,
        )
        self.y_model = tensor_ary  # pyright: ignore[reportAttributeAccessIssue]

        # NOTE: It's necessary to handle this here.
        self.memory_used += self.y_model.nbytes
        self.model.memory_used += self.y_model.nbytes

    def _sync_x_y(
        self, x_model: np.ndarray, y_model: np.ndarray
    ) -> tuple[TensorArray, TensorArray]:
        """Synchronize input and target batches to GPU memory."""
        # NOTE: in CUDA it's necessary to always have batches of the same size.
        local_batch_size = x_model.shape[0]

        if local_batch_size != 0:
            if local_batch_size != self.model.batch_size:
                # NOTE: if x_batch is empty (local_batch_size == 0), this will mean the
                # end of the loop where this function is called.
                num_repetitions = np.ceil(self.model.batch_size / local_batch_size)
                x_model = np.repeat(x_model, num_repetitions, axis=0)[: self.model.batch_size]
                y_model = np.repeat(y_model, num_repetitions, axis=0)[: self.model.batch_size]
            # else: The batch has the right shape ==> Nothing to do.

            x_model = np.asarray(x_model, dtype=self.model.dtype, order="C")
            y_model = np.asarray(y_model, dtype=self.model.dtype, order="C")

            assert isinstance(self.y, TensorArray) and isinstance(self.y_model, TensorArray)
            self.y.set(x_model)
            self.y_model.set(y_model)
            x, y_targ = self.model.layers[0].y, self.y_model
        else:
            empty_x = gpuarray.zeros((1, *self.model.dataset.input_shape), self.model.dtype)[:0]
            empty_y_tag = gpuarray.zeros((1, *self.model.dataset.output_shape), self.model.dtype)[
                :0
            ]
            x = TensorArray(empty_x, self.model.tensor_format, self.model.cudnn_dtype)
            y_targ = TensorArray(empty_y_tag, self.model.tensor_format, self.model.cudnn_dtype)
        return x, y_targ

    ws_size = 0
    ws: drv.DeviceAllocation = None

    @property
    def ws_ptr(self) -> ctypes.c_void_p:
        """Return the pointer to the workspace memory."""
        return ctypes.c_void_p(int(self.ws))

    def check_convolution_memory(self, size: int) -> None:
        """Allocate or reallocate workspace memory if required."""
        if size < self.ws_size:
            return

        if self.ws is not None:
            self.ws.free()

        self.ws_size = max(1, size)
        self.ws = drv.mem_alloc(self.ws_size)

    def get_convolution_workspace_ptr(self) -> ctypes.c_void_p:
        """Return the workspace pointer."""
        return self.ws_ptr

    def get_convolution_workspace_size(self) -> int:
        """Return the workspace size in bytes."""
        return self.ws_size
