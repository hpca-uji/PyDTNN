from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from pydtnn.model import Model
from pydtnn.layers.layer import Layer
from pydtnn.tracers.events import PYDTNN_MDL_EVENT, PYDTNN_MDL_EVENTS, PYDTNN_OPS_EVENT, PYDTNN_OPS_EVENTS, \
    PYDTNN_EVENT_FINISHED, PYDTNN_MDL_EVENT_enum, PYDTNN_OPS_EVENT_enum

try:
    from pydtnn.libs.libmpi import MPI
except Exception:
    pass
from pydtnn.utils.constants import ArrayShape
import numpy as np


class LayerCPU(Layer[np.ndarray]):
    """
    Extends a Layer class with the attributes and methods required by CPU Layers.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model: Model[np.ndarray]

    def initialize(self, prev_shape: ArrayShape, x: np.ndarray | None = None):
        super().initialize(prev_shape, x)

    @property
    def _ary_prop(self) -> set[str]:
        return {*self.grad_vars.keys(), *self.grad_vars.values()}

    def _export_prop(self, key: str):
        if key not in self._ary_prop:
            return super()._export_prop(key)

        ary = getattr(self, key)
        return np.asarray(ary, dtype=np.float64, order="C", copy=True)

    def _import_prop(self, key: str, value) -> None:
        if key not in self._ary_prop:
            return super()._import_prop(key, value)

        ary = getattr(self, key)
        ary[:] = np.asarray(value, dtype=self.model.dtype, order="C", copy=None)

    def reduce_weights_async(self, gradient=True):
        if not self.model.comm:
            return
        self.reqs_allred = {}

        for w_, dw_ in self.grad_vars.items():
            dw_ = dw_ if gradient else w_
            dw: np.ndarray = getattr(self, dw_)
            dw *= self.model.rank_weight
            if self.model.crypt:
                dw = self.model.crypt.encrypt(dw)  # type: ignore
            if self.model.use_mpi_buffers:
                req = self.model.comm.Iallreduce(MPI.IN_PLACE, dw, op=MPI.SUM)
            else:
                req = self.model.comm.iallreduce(dw, op=MPI.SUM)
            self.reqs_allred[dw_] = req

    def wait_allreduce_async(self, gradient=True):
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

    def reduce_weights_sync(self, gradient=True):
        if not self.model.comm:
            return
        for w_, dw_ in self.grad_vars.items():
            dw_ = dw_ if gradient else w_
            self.model.tracer.emit_nevent([PYDTNN_MDL_EVENT, PYDTNN_OPS_EVENT],
                                          [self.id * PYDTNN_MDL_EVENTS + PYDTNN_MDL_EVENT_enum.ALLREDUCE_DW,
                                           self.id * PYDTNN_OPS_EVENTS + PYDTNN_OPS_EVENT_enum.OPS_ALLREDUCE_DW])
            dw: np.ndarray = getattr(self, dw_)
            dw *= self.model.rank_weight
            if self.model.crypt:
                dw = self.model.crypt.encrypt(dw)  # type: ignore
            if self.model.use_mpi_buffers:
                self.model.comm.Allreduce(MPI.IN_PLACE, dw, op=MPI.SUM)
            else:
                dw = self.model.comm.allreduce(dw, op=MPI.SUM)
            if self.model.crypt:
                dw = self.model.crypt.decrypt(dw)  # type: ignore
            setattr(self, dw_, dw)
            self.model.tracer.emit_nevent([PYDTNN_MDL_EVENT, PYDTNN_OPS_EVENT], [PYDTNN_EVENT_FINISHED, PYDTNN_EVENT_FINISHED])
