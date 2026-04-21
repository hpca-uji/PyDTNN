import numpy as np

from pydtnn.abstract.layerable import Layerable
from pydtnn.backends.numpy.abstract.base import BaseNumpy
from pydtnn.tracers.events import PYDTNN_EVENT_FINISHED, PYDTNN_OPS_EVENT, PYDTNN_OPS_EVENTS, PYDTNN_OPS_EVENT_enum


class LayerableNumpy(Layerable[np.ndarray], BaseNumpy):

    def reduce_weights_async(self, gradient=True):
        # NOTE: Keep in sync with Layer
        if not self.model.comm:
            return
        assert len(self.reqs_allred) == 0, "MPI request overwritten (not waited)!"
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
        if not self.model.comm or self.model.enable_nccl or not self.reqs_allred:
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
        self.self.reqs_allred.clear()

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
