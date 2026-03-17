import logging
logger = logging.getLogger(__name__)

from pydtnn.backends.numpy.layers.layer import LayerNumpy
from pydtnn.layers.flatten import Flatten
from pydtnn.tracers.events import PYDTNN_OPS_EVENT, PYDTNN_OPS_EVENTS, PYDTNN_EVENT_FINISHED, PYDTNN_OPS_EVENT_enum

from pydtnn.libs import numpy as np
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    import numpy as np


class FlattenNumpy(Flatten[np.ndarray], LayerNumpy):

    def forward(self, x: np.ndarray) -> np.ndarray:
        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, self.id * PYDTNN_OPS_EVENTS + PYDTNN_OPS_EVENT_enum.FORWARD_RESHAPE_Y)
        y: np.ndarray = x.reshape((x.shape[0], *self.shape))
        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, PYDTNN_EVENT_FINISHED)
        return np.ascontiguousarray(y, dtype=self.model.dtype)
    # ---

    def backward(self, dy: np.ndarray) -> np.ndarray:
        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, self.id * PYDTNN_OPS_EVENTS + PYDTNN_OPS_EVENT_enum.BACKWARD_RESHAPE_DX)
        dx: np.ndarray = dy.reshape((dy.shape[0], *self.prev_shape))
        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, PYDTNN_EVENT_FINISHED)
        return np.ascontiguousarray(dx, dtype=self.model.dtype)
    # ---
