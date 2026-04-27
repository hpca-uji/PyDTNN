from pydtnn.abstract.layerable import Layerable
from pydtnn.backends.pycuda.abstract.base import BasePycuda
from pydtnn.backends.pycuda.utils.tensor_array import TensorArray
from pydtnn.tracers.events import (PYDTNN_EVENT_FINISHED, PYDTNN_OPS_EVENT,
                                   PYDTNN_OPS_EVENTS, PYDTNN_OPS_EVENT_enum)

try:
    import pydtnn.libs.nccl as nccl
except Exception as e:
    pass


class LayerablePycuda(Layerable[TensorArray], BasePycuda):

    def reduce_weights_async(self, gradient=True):
        # NOTE: Keep in sync with Layer
        if not self.model.comm:
            return

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
                #             dw.get_async(self.stream_2, dw_cpu)
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

                assert dw_ not in self.reqs_allred, f"MPI request overwritten ({dw_} not waited)!"
                req = self.model._layer_reduce_async(dw_cpu)
                self.reqs_allred[dw_] = req

    def wait_allreduce_async(self, gradient=True):
        # NOTE: Keep in sync with Layer
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
                req = self.reqs_allred.pop(dw_, None)
                if req is None:
                    continue  # noqa: E701
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
                #                 dw.set_async(dw_cpu, self.stream_2)
                #
                #         nccl.ncclBroadcast(dw.ptr, dw.ptr, dw.size, self.model.nccl_type,
                #                            root=0, comm=self.model.nccl_comm,
                #                            stream=self.stream_2.handle)

                dw = getattr(self, dw_)
                dw_cpu = getattr(self, f"{dw_}_cpu")

                # If there is no CUDA-aware MPI, copy data back to GPU
                dw.set_async(dw_cpu, self.stream_2)

    def reduce_weights_sync(self, gradient=True):
        # NOTE: Keep in sync with Layer
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
                #             dw_cpu = dw.get()
                #             self.model.inter_comm.Allreduce(MPI.IN_PLACE, dw_cpu, op=MPI.SUM)
                #             dw.set_async(dw_cpu, self.stream_2)
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
                dw.set_async(dw_cpu, self.stream_2)
