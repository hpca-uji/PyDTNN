from pydtnn.activations.activation import Activation
from pydtnn.tracers.events import PYDTNN_MDL_EVENT, PYDTNN_MDL_EVENTS, PYDTNN_OPS_EVENT, PYDTNN_OPS_EVENTS, \
    PYDTNN_EVENT_FINISHED, PYDTNN_MDL_EVENT_enum, PYDTNN_OPS_EVENT_enum

try:
    from pydtnn.libs.libmpi import MPI
except Exception:
    pass
import numpy as np


class ActivationCPU(Activation[np.ndarray]):
    """
    Extends an Activation class with the attributes and methods required by CPU Activations.

    The next methods are copied from LayerCPU:
      * reduce_weights_async()
      * wait_allreduce_async()
      * reduce_weights_sync()
    """

    def initialize(self, prev_shape, x: np.ndarray | None = None):
        super().initialize(prev_shape, x)

    def reduce_weights_async(self, gradient: bool = True) -> None:
        if not self.model.comm:
            return
        self.reqs_allred = {}

        for w_, dw_ in self.grad_vars.items():
            dw_ = dw_ if gradient else w_
            dw: np.ndarray = getattr(self, dw_)
            np.multiply(dw, self.model.rank_weight, out=dw,
                        dtype=self.model.dtype)
            if self.model.crypt:
                dw = self.model.crypt.encrypt(dw)
            if self.model.use_mpi_buffers:
                req = self.model.comm.Iallreduce(MPI.IN_PLACE, dw, op=MPI.SUM)
            else:
                req = self.model.comm.iallreduce(dw, op=MPI.SUM)
            self.reqs_allred[dw_] = req

    def wait_allreduce_async(self, gradient=True) -> None:
        if not self.model.comm or self.model.enable_nccl:
            return
        for w_, dw_ in self.grad_vars.items():
            dw_ = dw_ if gradient else w_
            res = self.reqs_allred[dw_].wait()
            if res is None:
                dw = getattr(self, dw_)
            else:
                dw = res
            if self.model.crypt:
                dw = self.model.crypt.decrypt(dw)
            setattr(self, dw_, dw)

    def reduce_weights_sync(self, gradient=True) -> None:
        if not self.model.comm:
            return
        for w_, dw_ in self.grad_vars.items():
            dw_ = dw_ if gradient else w_
            self.model.tracer.emit_nevent([PYDTNN_MDL_EVENT, PYDTNN_OPS_EVENT],
                                          [self.id * PYDTNN_MDL_EVENTS + PYDTNN_MDL_EVENT_enum.ALLREDUCE_DW,
                                           self.id * PYDTNN_OPS_EVENTS + PYDTNN_OPS_EVENT_enum.OPS_ALLREDUCE_DW])
            dw: np.ndarray = getattr(self, dw_)
            np.multiply(dw, self.model.rank_weight, out=dw,
                        dtype=self.model.dtype)
            if self.model.crypt:
                dw = self.model.crypt.encrypt(dw)  #type: ignore
            if self.model.use_mpi_buffers:
                self.model.comm.Allreduce(MPI.IN_PLACE, dw, op=MPI.SUM)
            else:
                dw = self.model.comm.allreduce(dw, op=MPI.SUM)
            if self.model.crypt:
                dw = self.model.crypt.decrypt(dw)  #type: ignore
            setattr(self, dw_, dw)
            self.model.tracer.emit_nevent([PYDTNN_MDL_EVENT, PYDTNN_OPS_EVENT], [PYDTNN_EVENT_FINISHED, PYDTNN_EVENT_FINISHED])
