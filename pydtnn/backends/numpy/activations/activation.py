from typing import TYPE_CHECKING
from pydtnn.backends.numpy.abstract.layerable import LayerableNumpy
from pydtnn.libs import numpy as np
from pydtnn.tracers.events import PYDTNN_MDL_EVENT, PYDTNN_MDL_EVENTS, PYDTNN_OPS_EVENT, PYDTNN_OPS_EVENTS, \
    PYDTNN_EVENT_FINISHED, PYDTNN_MDL_EVENT_enum, PYDTNN_OPS_EVENT_enum
from pydtnn.activations.activation import Activation
import logging
logger = logging.getLogger(__name__)


try:
    from pydtnn.libs.mpi import MPI
except Exception:
    pass
if TYPE_CHECKING:
    import numpy as np


class ActivationNumpy(Activation[np.ndarray], LayerableNumpy):
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
            self.model.tracer.emit_event(PYDTNN_OPS_EVENT, self.id * PYDTNN_OPS_EVENTS + PYDTNN_OPS_EVENT_enum.LAYER_ENCODE)
            dw = self.model._layer_reduce_encode(dw)
            self.model.tracer.emit_event(PYDTNN_OPS_EVENT, PYDTNN_EVENT_FINISHED)
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
            self.model.tracer.emit_event(PYDTNN_OPS_EVENT, self.id * PYDTNN_OPS_EVENTS + PYDTNN_OPS_EVENT_enum.LAYER_DECODE)
            dw = self.model._layer_reduce_decode(dw)
            self.model.tracer.emit_event(PYDTNN_OPS_EVENT, PYDTNN_EVENT_FINISHED)
            setattr(self, dw_, dw)

    def reduce_weights_sync(self, gradient=True):
        # NOTE: Keep in sync with Layer
        if not self.model.comm:
            return
        for w_, dw_ in self.grad_vars.items():
            dw_ = dw_ if gradient else w_
            dw: np.ndarray = getattr(self, dw_)

            self.model.tracer.emit_event(PYDTNN_OPS_EVENT, self.id * PYDTNN_OPS_EVENTS + PYDTNN_OPS_EVENT_enum.LAYER_ENCODE)
            dw = self.model._layer_reduce_encode(dw)
            self.model.tracer.emit_event(PYDTNN_OPS_EVENT, PYDTNN_EVENT_FINISHED)

            self.model.tracer.emit_event(PYDTNN_OPS_EVENT, self.id * PYDTNN_OPS_EVENTS + PYDTNN_OPS_EVENT_enum.OPS_ALLREDUCE_DW)
            dw = self.model._layer_reduce_sync(dw)
            self.model.tracer.emit_event(PYDTNN_OPS_EVENT, PYDTNN_EVENT_FINISHED)

            self.model.tracer.emit_event(PYDTNN_OPS_EVENT, self.id * PYDTNN_OPS_EVENTS + PYDTNN_OPS_EVENT_enum.LAYER_DECODE)
            dw = self.model._layer_reduce_decode(dw)
            self.model.tracer.emit_event(PYDTNN_OPS_EVENT, PYDTNN_EVENT_FINISHED)

            setattr(self, dw_, dw)
