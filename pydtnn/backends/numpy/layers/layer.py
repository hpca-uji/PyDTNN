import logging
logger = logging.getLogger(__name__)

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from pydtnn.model import Model
from pydtnn.layers.layer import Layer
from pydtnn.tracers.events import PYDTNN_MDL_EVENT, PYDTNN_MDL_EVENTS, PYDTNN_OPS_EVENT, PYDTNN_OPS_EVENTS, \
    PYDTNN_EVENT_FINISHED, PYDTNN_MDL_EVENT_enum, PYDTNN_OPS_EVENT_enum

try:
    from pydtnn.libs.mpi import MPI
except Exception:
    pass
from pydtnn.utils.constants import ArrayShape
from pydtnn.libs import numpy as np
if TYPE_CHECKING:
    import numpy as np


class LayerNumpy(Layer[np.ndarray]):
    """
    Extends a Layer class with the attributes and methods required by CPU Layers.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model: Model[np.ndarray]

    def _model_init(self, prev_shape: ArrayShape, x: np.ndarray | None = None):
        super()._model_init(prev_shape, x)

    @property
    def _ary_prop(self) -> set[str]:
        return {*self.grad_vars.keys(), *self.grad_vars.values()}

    def _export_prop(self, key: str):
        if key not in self._ary_prop:
            return super()._export_prop(key)

        ary = getattr(self, key)
        return np.asarray(ary, dtype=np.float64, order="C").copy()

    def _import_prop(self, key: str, value) -> None:
        if key not in self._ary_prop:
            return super()._import_prop(key, value)

        ary = getattr(self, key)
        ary[:] = np.asarray(value, dtype=self.model.dtype, order="C")

    def reduce_weights_async(self, gradient=True):
        # NOTE: Keep in sync with Activation
        if not self.model.comm:
            return
        self.reqs_allred = {}

        for w_, dw_ in self.grad_vars.items():
            dw_ = dw_ if gradient else w_
            dw: np.ndarray = getattr(self, dw_)
            dw = self.model._layer_reduce_encode(dw)
            req = self.model._layer_reduce_async(dw)
            self.reqs_allred[dw_] = req

    def wait_allreduce_async(self, gradient=True):
        # NOTE: Keep in sync with Activation
        if not self.model.comm or self.model.enable_nccl:
            return
        for w_, dw_ in self.grad_vars.items():
            dw_ = dw_ if gradient else w_
            dw = getattr(self, dw_)
            req = self.reqs_allred[dw_]
            dw = self.model._layer_reduce_wait(dw, req)
            dw = self.model._layer_reduce_decode(dw)
            setattr(self, dw_, dw)

    def reduce_weights_sync(self, gradient=True):
        # NOTE: Keep in sync with Activation
        if not self.model.comm:
            return
        for w_, dw_ in self.grad_vars.items():
            dw_ = dw_ if gradient else w_
            self.model.tracer.emit_nevent([PYDTNN_MDL_EVENT, PYDTNN_OPS_EVENT],
                                          [self.id * PYDTNN_MDL_EVENTS + PYDTNN_MDL_EVENT_enum.ALLREDUCE_DW,
                                           self.id * PYDTNN_OPS_EVENTS + PYDTNN_OPS_EVENT_enum.OPS_ALLREDUCE_DW])
            dw: np.ndarray = getattr(self, dw_)
            dw = self.model._layer_reduce_encode(dw)
            dw = self.model._layer_reduce_sync(dw)
            dw = self.model._layer_reduce_decode(dw)
            setattr(self, dw_, dw)
            self.model.tracer.emit_nevent([PYDTNN_MDL_EVENT, PYDTNN_OPS_EVENT], [PYDTNN_EVENT_FINISHED, PYDTNN_EVENT_FINISHED])

    def _sync_x_y(self, x_batch: np.ndarray, y_batch: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        x_batch = np.asarray(x_batch, dtype=self.model.dtype, order="C")
        y_batch = np.asarray(y_batch, dtype=self.model.dtype, order="C")
        return x_batch, y_batch
