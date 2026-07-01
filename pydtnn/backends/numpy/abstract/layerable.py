"""
Numpy backend implementation for layerable components in PyDTNN.
"""

import numpy as np

from pydtnn.abstract.layerable import Layerable
from pydtnn.backends.numpy.abstract.base import BaseNumpy
from pydtnn.tracers.events import (PYDTNN_EVENT_FINISHED, PYDTNN_OPS_EVENT,
                                   PYDTNN_OPS_EVENTS, OpsEventEnum)

__all__ = ("LayerableNumpy",)


class LayerableNumpy(Layerable[np.ndarray], BaseNumpy):
    """
    Numpy-specific implementation of a layerable component supporting distributed operations.
    """

    def reduce_weights_async(self, gradient: bool = True) -> None:
        """
        Initiates an asynchronous all-reduce operation for weights or gradients.

        Args:
            gradient (bool): If True, reduces gradients; otherwise, reduces weights.
        """
        # NOTE: Keep in sync with Layer
        if not self.model.comm:
            return

        for w_, dw_ in self.grad_vars.items():
            dw_ = dw_ if gradient else w_
            dw: np.ndarray = getattr(self, dw_)
            self.model.tracer.emit_event(
                PYDTNN_OPS_EVENT, self.id * PYDTNN_OPS_EVENTS + OpsEventEnum.LAYER_ENCODE
            )
            dw = self.model._layer_reduce_encode(dw)
            self.model.tracer.emit_event(PYDTNN_OPS_EVENT, PYDTNN_EVENT_FINISHED)
            assert dw_ not in self.reqs_allred, f"MPI request overwritten ({dw_} not waited)!"
            req = self.model._layer_reduce_async(dw)
            self.reqs_allred[dw_] = req

    def wait_allreduce_async(self, gradient: bool = True) -> None:
        """
        Waits for completion of asynchronous all-reduce operations and decodes results.

        Args:
            gradient (bool): If True, waits for gradients; otherwise, waits for weights.
        """
        # NOTE: Keep in sync with Layer
        if not self.model.comm or self.model.enable_nccl:
            return
        for w_, dw_ in self.grad_vars.items():
            dw_ = dw_ if gradient else w_
            dw = getattr(self, dw_)
            req = self.reqs_allred.pop(dw_, None)
            if req is None:
                continue  # noqa: E701
            dw = self.model._layer_reduce_wait(dw, req)
            self.model.tracer.emit_event(
                PYDTNN_OPS_EVENT, self.id * PYDTNN_OPS_EVENTS + OpsEventEnum.LAYER_DECODE
            )
            dw = self.model._layer_reduce_decode(dw)
            self.model.tracer.emit_event(PYDTNN_OPS_EVENT, PYDTNN_EVENT_FINISHED)
            setattr(self, dw_, dw)

    def reduce_weights_sync(self, gradient: bool = True) -> None:
        """
        Performs a synchronous all-reduce operation for weights or gradients.

        Args:
            gradient (bool): If True, reduces gradients; otherwise, reduces weights.
        """
        # NOTE: Keep in sync with Layer
        if not self.model.comm:
            return
        for w_, dw_ in self.grad_vars.items():
            dw_ = dw_ if gradient else w_
            dw: np.ndarray = getattr(self, dw_)

            self.model.tracer.emit_event(
                PYDTNN_OPS_EVENT, self.id * PYDTNN_OPS_EVENTS + OpsEventEnum.LAYER_ENCODE
            )
            dw = self.model._layer_reduce_encode(dw)
            self.model.tracer.emit_event(PYDTNN_OPS_EVENT, PYDTNN_EVENT_FINISHED)

            self.model.tracer.emit_event(
                PYDTNN_OPS_EVENT,
                self.id * PYDTNN_OPS_EVENTS + OpsEventEnum.OPS_ALLREDUCE_DW,
            )
            dw = self.model._layer_reduce_sync(dw)
            self.model.tracer.emit_event(PYDTNN_OPS_EVENT, PYDTNN_EVENT_FINISHED)

            self.model.tracer.emit_event(
                PYDTNN_OPS_EVENT, self.id * PYDTNN_OPS_EVENTS + OpsEventEnum.LAYER_DECODE
            )
            dw = self.model._layer_reduce_decode(dw)
            self.model.tracer.emit_event(PYDTNN_OPS_EVENT, PYDTNN_EVENT_FINISHED)

            setattr(self, dw_, dw)
