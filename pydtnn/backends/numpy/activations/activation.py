import logging
logger = logging.getLogger(__name__)

from pydtnn.activations.activation import Activation
from pydtnn.tracers.events import PYDTNN_MDL_EVENT, PYDTNN_MDL_EVENTS, PYDTNN_OPS_EVENT, PYDTNN_OPS_EVENTS, \
    PYDTNN_EVENT_FINISHED, PYDTNN_MDL_EVENT_enum, PYDTNN_OPS_EVENT_enum

try:
    from pydtnn.libs.mpi import MPI
except Exception:
    pass
from pydtnn.libs import numpy as np
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    import numpy as np


class ActivationNumpy(Activation[np.ndarray]):
    """
    Extends an Activation class with the attributes and methods required by CPU Activations.

    The next methods are copied from LayerNumpy:
      * reduce_weights_async()
      * wait_allreduce_async()
      * reduce_weights_sync()
    """

    def _model_init(self, prev_shape, x: np.ndarray | None = None):
        super()._model_init(prev_shape, x)

    def reduce_weights_async(self, gradient=True):
        # NOTE: Keep in sync with Layer
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
        # NOTE: Keep in sync with Layer
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
        # NOTE: Keep in sync with Layer
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
