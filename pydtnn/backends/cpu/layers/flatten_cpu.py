from pydtnn.backends.cpu.layers import LayerCPU
from pydtnn.layers import Flatten
from pydtnn.tracers import PYDTNN_OPS_EVENT, PYDTNN_OPS_EVENTS, PYDTNN_EVENT_FINISHED, PYDTNN_OPS_EVENT_enum

from numpy import ndarray
class FlattenCPU(LayerCPU, Flatten):

    def forward(self, x:ndarray) -> ndarray:
        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, self.id * PYDTNN_OPS_EVENTS + PYDTNN_OPS_EVENT_enum.FORWARD_RESHAPE_Y)
        y:ndarray = x.reshape((x.shape[0], *self.shape), order="C", copy=None)
        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, PYDTNN_EVENT_FINISHED)
        return y

    def backward(self, dy:ndarray) -> ndarray:
        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, self.id * PYDTNN_OPS_EVENTS + PYDTNN_OPS_EVENT_enum.BACKWARD_RESHAPE_DX)
        dx:ndarray = dy.reshape((dy.shape[0], *self.prev_shape), order="C", copy=None)
        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, PYDTNN_EVENT_FINISHED)
        return dx
