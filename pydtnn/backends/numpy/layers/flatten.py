"""Numpy backend implementation of the Flatten layer."""

import logging
from typing import TYPE_CHECKING

from pydtnn.backends.numpy.layers.abstract.layer import LayerNumpy
from pydtnn.layers.flatten import Flatten
from pydtnn.libs import numpy as np
from pydtnn.tracers.events import (PYDTNN_EVENT_FINISHED, PYDTNN_OPS_EVENT,
                                   PYDTNN_OPS_EVENTS, OpsEventEnum)

__all__ = ("FlattenNumpy",)

logger = logging.getLogger(__name__)


if TYPE_CHECKING:
    import numpy as np


class FlattenNumpy(Flatten[np.ndarray], LayerNumpy):
    """Numpy-based Flatten layer implementation."""

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Flattens the input tensor while preserving the batch dimension.

        Args:
            x: Input tensor of shape (batch_size, ...).

        Returns:
            Flattened tensor of shape (batch_size, *self.shape).
        """
        self.model.tracer.emit_event(
            PYDTNN_OPS_EVENT, self.id * PYDTNN_OPS_EVENTS + OpsEventEnum.FORWARD_RESHAPE_Y
        )
        y: np.ndarray = x.reshape((x.shape[0], *self.shape))
        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, PYDTNN_EVENT_FINISHED)
        return np.asarray(y, dtype=self.model.dtype, order="C")

    def backward(self, dy: np.ndarray) -> np.ndarray:
        """
        Reshapes the gradient back to the original input shape.

        Args:
            dy: Gradient of the loss with respect to the output.

        Returns:
            Gradient of the loss with respect to the input.
        """
        self.model.tracer.emit_event(
            PYDTNN_OPS_EVENT,
            self.id * PYDTNN_OPS_EVENTS + OpsEventEnum.BACKWARD_RESHAPE_DX,
        )
        dx: np.ndarray = dy.reshape((dy.shape[0], *self.prev_shape))
        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, PYDTNN_EVENT_FINISHED)
        return np.asarray(dx, dtype=self.model.dtype, order="C")
