import logging
logger = logging.getLogger(__name__)

import numpy as np

from pydtnn.layers.layer import Layer
from pydtnn.backends.pycuda.abstract.layerable import LayerablePycuda
from pydtnn.tracers.events import PYDTNN_MDL_EVENT, PYDTNN_MDL_EVENTS, PYDTNN_OPS_EVENT, PYDTNN_OPS_EVENTS, \
    PYDTNN_EVENT_FINISHED, PYDTNN_MDL_EVENT_enum, PYDTNN_OPS_EVENT_enum

try:
    from pydtnn.libs.mpi import MPI
except Exception as e:
    pass

try:
    import pydtnn.libs.nccl as nccl
except Exception as e:
    pass

from numpy import ndarray
from pydtnn import gpu_errors
from pydtnn.backends.pycuda.utils.tensor_array import TensorArray

from pycuda import gpuarray  # type: ignore


class LayerPycuda(Layer[TensorArray], LayerablePycuda):
    """
    Extends a Layer class with the attributes and methods required by GPU Layers.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # GPU layer attributes
        # NOTE: All of these values will be initalized in the "initialize" method.
        self.weights_cpu: ndarray = None  # type: ignore
        self.biases_cpu: ndarray = None  # type: ignore
        self.dx: TensorArray = None  # type: ignore
        self.dw: TensorArray = None  # type: ignore
        self.db: TensorArray = None  # type: ignore
        self.dw_cpu: ndarray = None  # type: ignore
        self.db_cpu: ndarray = None  # type: ignore
        self.one_vec_cpu: ndarray = None  # type: ignore
        self.one_vec_gpu: gpuarray.GPUArray = None  # type: ignore
        self.grid: tuple[int, int, int] = None  # type: ignore
        self.block: tuple[int, int, int] = None  # type: ignore

    def _model_init(self, prev_shape: tuple[int, ...], x: TensorArray | None = None) -> None:
        super()._model_init(prev_shape, x)

        if not self.model.enable_cudnn:
            raise ExceptionGroup("GPU layers requires CUDNN to be enabled!", gpu_errors)

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
        cpu_ary = np.asarray(gpu_ary.get(), dtype=np.float64, order="C").copy()
        return cpu_ary

    def _import_prop(self, key: str, value) -> None:
        if key not in self._ary_prop:
            return super()._import_prop(key, value)

        gpu_ary = getattr(self, key).ary
        cpu_ary = np.asarray(value.reshape(gpu_ary.shape), dtype=self.model.dtype, order="C")
        gpu_ary.set(cpu_ary)

    def reduce_weights_async(self, gradient=True):
        # NOTE: Keep in sync with Activation
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
                # self.model.stream.synchronize()
                dw *= self.model.rank_weight
                # TODO: self.model._encode_reduce
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

                dw_cpu = getattr(self, f"{dw_}_cpu")
                self.model.tracer.emit_event(PYDTNN_OPS_EVENT, self.id * PYDTNN_OPS_EVENTS + PYDTNN_OPS_EVENT_enum.LAYER_ENCODE)
                dw_cpu = self.model._layer_reduce_encode(dw_cpu)
                self.model.tracer.emit_event(PYDTNN_OPS_EVENT, PYDTNN_EVENT_FINISHED)
                req = self.model._layer_reduce_async(dw_cpu)
                self.reqs_allred[dw_] = req

    def wait_allreduce_async(self, gradient=True):
        # NOTE: Keep in sync with Activation
        if not self.model.comm:
            return

        for w_, dw_ in self.grad_vars.items():
            if self.model.enable_nccl:
                # self.model.stream.synchronize()
                dw: TensorArray = getattr(self, dw_)
                # TODO: self.model._decode_reduce
                setattr(self, dw_, dw)
            else:
                dw_ = dw_ if gradient else w_
                dw_cpu = getattr(self, f"{dw_}_cpu")
                req = self.reqs_allred[dw_]
                dw_cpu = self.model._layer_reduce_wait(dw_cpu, req)
                self.model.tracer.emit_event(PYDTNN_OPS_EVENT, self.id * PYDTNN_OPS_EVENTS + PYDTNN_OPS_EVENT_enum.LAYER_DECODE)
                dw_cpu = self.model._layer_reduce_decode(dw_cpu)  # FIXME: dw and dw_cpu relation unclear
                self.model.tracer.emit_event(PYDTNN_OPS_EVENT, PYDTNN_EVENT_FINISHED)
                setattr(self, f"{dw_}_cpu", dw_cpu)

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

                dw = getattr(self, dw_)
                dw_cpu = getattr(self, f"{dw_}_cpu")

                # If there is no CUDA-aware MPI, copy data back to GPU
                dw.ary.set_async(dw_cpu, self.stream_2)

    def reduce_weights_sync(self, gradient=True):
        # NOTE: Keep in sync with Activation
        if not self.model.comm:
            return

        for w_, dw_ in self.grad_vars.items():
            dw_ = dw_ if gradient else w_
            # stream = self.stream_2.handle)
            dw = getattr(self, dw_)

            if self.model.enable_nccl:
                # self.stream_2.synchronize()
                dw *= self.model.rank_weight
                # TODO: self.model._encode_reduce
                self.model.tracer.emit_event(PYDTNN_OPS_EVENT, self.id * PYDTNN_OPS_EVENTS + PYDTNN_OPS_EVENT_enum.OPS_ALLREDUCE_DW)
                nccl.ncclAllReduce(dw.ptr, dw.ptr, dw.size, self.model.nccl_type,
                                   nccl.RedOp.Sum, comm=self.model.nccl_comm,
                                   stream=self.stream_2.handle)
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

                self.model.tracer.emit_event(PYDTNN_OPS_EVENT, self.id * PYDTNN_OPS_EVENTS + PYDTNN_OPS_EVENT_enum.LAYER_ENCODE)
                dw_cpu = self.model._layer_reduce_encode(dw_cpu)
                self.model.tracer.emit_event(PYDTNN_OPS_EVENT, PYDTNN_EVENT_FINISHED)

                self.model.tracer.emit_event(PYDTNN_OPS_EVENT, self.id * PYDTNN_OPS_EVENTS + PYDTNN_OPS_EVENT_enum.OPS_ALLREDUCE_DW)
                dw_cpu = self.model._layer_reduce_sync(dw_cpu)
                self.model.tracer.emit_event(PYDTNN_OPS_EVENT, PYDTNN_EVENT_FINISHED)

                self.model.tracer.emit_event(PYDTNN_OPS_EVENT, self.id * PYDTNN_OPS_EVENTS + PYDTNN_OPS_EVENT_enum.LAYER_DECODE)
                dw_cpu = self.model._layer_reduce_decode(dw_cpu)
                self.model.tracer.emit_event(PYDTNN_OPS_EVENT, PYDTNN_EVENT_FINISHED)

                setattr(self, f"{dw_}_cpu", dw_cpu)

                # If there is no CUDA-aware MPI, copy data back to GPU
                dw.ary.set_async(dw_cpu, self.stream_2)

    def _sync_x_y(self, x_batch: np.ndarray, y_batch: np.ndarray) -> tuple[TensorArray, TensorArray]:
        # NOTE: in CUDA it's necessary to always have batches of the same size.
        local_batch_size = x_batch.shape[0]

        if local_batch_size != 0:
            if local_batch_size != self.model.batch_size:
                # NOTE: if x_batch is empty (local_batch_size == 0), this will mean the end of the loop where this function is called.
                num_repetitions = np.ceil(self.model.batch_size / local_batch_size)
                x_batch = np.repeat(x_batch, num_repetitions, axis=0)[:self.model.batch_size]
                y_batch = np.repeat(y_batch, num_repetitions, axis=0)[:self.model.batch_size]
            # else: The batch has the right shape ==> Nothing to do.

            x_batch = np.asarray(x_batch, dtype=self.model.dtype, order="C")
            y_batch = np.asarray(y_batch, dtype=self.model.dtype, order="C")

            assert isinstance(self.y, TensorArray) and isinstance(self.model.y_batch, TensorArray)
            self.y.ary.set(x_batch)
            self.model.y_batch.ary.set(y_batch)
            x, y_targ = self.model.layers[0].y, self.model.y_batch
        else:
            empty_x = gpuarray.zeros((1, *self.model.dataset.input_shape), self.model.dtype)[:0]
            empty_y_tag = gpuarray.zeros((1, *self.model.dataset.output_shape), self.model.dtype)[:0]
            x = TensorArray(empty_x, self.model.tensor_format, self.model.cudnn_dtype)
            y_targ = TensorArray(empty_y_tag, self.model.tensor_format, self.model.cudnn_dtype)
        return x, y_targ
