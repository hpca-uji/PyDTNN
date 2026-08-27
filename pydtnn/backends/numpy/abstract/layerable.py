"""Numpy backend implementation for layerable components in PyDTNN."""

import numpy as np

from pydtnn.abstract.layerable import Layerable
from pydtnn.backends.numpy.abstract.base import BaseNumpy
from pydtnn.tracers.events import (PYDTNN_EVENT_FINISHED, PYDTNN_OPS_EVENT,
                                   PYDTNN_OPS_EVENTS, OpsEventEnum)

__all__ = ("LayerableNumpy",)


class LayerableNumpy(Layerable[np.ndarray], BaseNumpy):
    """Numpy-specific implementation of a layerable component supporting distributed operations."""

    def reduce_weights_async(self, gradient: bool = True) -> None:
        """
        Initiates an asynchronous all-reduce operation for weights or gradients.

        Args:
            gradient (bool): If True, reduces gradients; otherwise, reduces weights.
        """
        # NOTE: Keep in sync with Layer
        if not self.model.comm:
            return

        # NOTE:  self.grad_vars = {[VAR]: [VAR's GRADIENT]}
        vars_to_iterate = self.grad_vars.values() if gradient else self.grad_vars.keys()

        for w_dw in vars_to_iterate:
            self._reduce_weights_sync(w_dw)

    def _reduce_weights_async(self, weights_: str) -> None:
        weights: np.ndarray = getattr(self, weights_)
        self.model.tracer.emit_event(
            PYDTNN_OPS_EVENT, self.id * PYDTNN_OPS_EVENTS + OpsEventEnum.LAYER_ENCODE
        )
        weights = self.model._layer_reduce_encode(weights)
        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, PYDTNN_EVENT_FINISHED)
        assert weights_ not in self.reqs_allred, f"MPI request overwritten ({weights_} not waited)!"
        req = self.model._layer_reduce_async(weights)
        self.reqs_allred[weights_] = req

    def wait_allreduce_async(self, gradient: bool = True) -> None:
        """
        Waits for completion of asynchronous all-reduce operations and decodes results.

        Args:
            gradient (bool): If True, waits for gradients; otherwise, waits for weights.
        """
        # NOTE: Keep in sync with Layer
        #if not self.model.comm or self.model.use_nccl:
        if not self.model.comm:
            return

        # NOTE:  self.grad_vars = {[VAR]: [VAR's GRADIENT]}
        vars_to_iterate = self.grad_vars.values() if gradient else self.grad_vars.keys()
        for w_dw in vars_to_iterate:
            self._wait_allreduce_async(w_dw)

    def _wait_allreduce_async(self, weights_: str) -> None:
        weights = getattr(self, weights_)
        req = self.reqs_allred.pop(weights_, None)
        if req is None:
            return
        weights = self.model._layer_reduce_wait(weights, req)
        self.model.tracer.emit_event(
            PYDTNN_OPS_EVENT, self.id * PYDTNN_OPS_EVENTS + OpsEventEnum.LAYER_DECODE
        )
        weights = self.model._layer_reduce_decode(weights)
        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, PYDTNN_EVENT_FINISHED)
        setattr(self, weights_, weights)

    def reduce_weights_sync(self, gradient: bool = True) -> None:
        """
        Performs a synchronous all-reduce operation for weights or gradients.

        Args:
            gradient (bool): If True, reduces gradients; otherwise, reduces weights.
        """
        # NOTE: Keep in sync with Layer
        if not self.model.comm:
            return

        # NOTE:  self.grad_vars = {[VAR]: [VAR's GRADIENT]}
        vars_to_iterate = self.grad_vars.values() if gradient else self.grad_vars.keys()

        for w_dw in vars_to_iterate:
            self._reduce_weights_sync(w_dw)

    def _reduce_weights_sync(self, weights_: str) -> None:
        weights: np.ndarray = getattr(self, weights_)
        
        self.model.tracer.emit_event(
            PYDTNN_OPS_EVENT, self.id * PYDTNN_OPS_EVENTS + OpsEventEnum.LAYER_ENCODE
        )
        weights = self.model._layer_reduce_encode(weights)
        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, PYDTNN_EVENT_FINISHED)

        self.model.tracer.emit_event(
            PYDTNN_OPS_EVENT,
            self.id * PYDTNN_OPS_EVENTS + OpsEventEnum.OPS_ALLREDUCE_DW,
        )
        weights = self.model._layer_reduce_sync(weights)
        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, PYDTNN_EVENT_FINISHED)

        self.model.tracer.emit_event(
            PYDTNN_OPS_EVENT, self.id * PYDTNN_OPS_EVENTS + OpsEventEnum.LAYER_DECODE
        )
        weights = self.model._layer_reduce_decode(weights)
        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, PYDTNN_EVENT_FINISHED)

        setattr(self, weights_, weights)
