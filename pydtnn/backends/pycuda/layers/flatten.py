"""PyCUDA implementation of the Flatten layer."""

import logging

from pydtnn.backends.pycuda.layers.abstract.layer import LayerPycuda
from pydtnn.backends.pycuda.utils.tensor_array import TensorArray
from pydtnn.layers.flatten import Flatten
from pydtnn.tracers.events import (PYDTNN_EVENT_FINISHED, PYDTNN_OPS_EVENT,
                                   PYDTNN_OPS_EVENTS, OpsEventEnum)

__all__ = ("FlattenPycuda",)

logger = logging.getLogger(__name__)


class FlattenPycuda(Flatten[TensorArray], LayerPycuda):
    """PyCUDA-accelerated Flatten layer implementation."""

    def _model_init(self, prev_shape, x):
        """Initialize layer parameters and output reference."""
        super()._model_init(prev_shape, x)
        self.y = x  # type: ignore (it's okay)

    def forward(self, x: TensorArray) -> TensorArray:
        """Perform forward pass by reshaping the input tensor."""
        self.model.tracer.emit_event(
            PYDTNN_OPS_EVENT, self.id * PYDTNN_OPS_EVENTS + OpsEventEnum.FORWARD_RESHAPE_Y
        )
        self.y = x.reshape((self.model.batch_size, *self.shape))
        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, PYDTNN_EVENT_FINISHED)
        return self.y

    def backward(self, dy: TensorArray) -> TensorArray:
        """Perform backward pass by reshaping the gradient tensor."""
        self.model.tracer.emit_event(
            PYDTNN_OPS_EVENT,
            self.id * PYDTNN_OPS_EVENTS + OpsEventEnum.BACKWARD_RESHAPE_DX,
        )
        self.dx = dy.reshape((self.model.batch_size, *self.prev_shape))
        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, PYDTNN_EVENT_FINISHED)
        return self.dx
