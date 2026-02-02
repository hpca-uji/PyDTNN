from pydtnn.backends.cupy.layers.layer import LayerCUPY
from pydtnn.layers.flatten import Flatten
from pydtnn.tracers.events import PYDTNN_OPS_EVENT, PYDTNN_OPS_EVENTS, PYDTNN_EVENT_FINISHED, PYDTNN_OPS_EVENT_enum

import cupy as np


class FlattenCUPY(Flatten[np.ndarray], LayerCUPY):

    def forward(self, x: np.ndarray) -> np.ndarray:
        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, self.id * PYDTNN_OPS_EVENTS + PYDTNN_OPS_EVENT_enum.FORWARD_RESHAPE_Y)
        y: np.ndarray = x.reshape((x.shape[0], *self.shape))
        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, PYDTNN_EVENT_FINISHED)
        return y
    # ---

    def backward(self, dy: np.ndarray) -> np.ndarray:
        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, self.id * PYDTNN_OPS_EVENTS + PYDTNN_OPS_EVENT_enum.BACKWARD_RESHAPE_DX)
        dx: np.ndarray = dy.reshape((dy.shape[0], *self.prev_shape))
        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, PYDTNN_EVENT_FINISHED)
        return dx
    # ---
