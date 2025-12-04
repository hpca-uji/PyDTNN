from pydtnn.layers.flatten import Flatten
from pydtnn.tracers.events import PYDTNN_OPS_EVENT, PYDTNN_OPS_EVENTS, PYDTNN_EVENT_FINISHED, PYDTNN_OPS_EVENT_enum
from pydtnn.backends.gpu.layers.layer import LayerGPU
from pydtnn.backends.gpu.utils.tensor_gpu import TensorGPU


class FlattenGPU(Flatten[TensorGPU], LayerGPU):

    def initialize(self, prev_shape, x):
        super().initialize(prev_shape, x)
        self.y = x  # type: ignore (it's okay)

    def forward(self, x: TensorGPU) -> TensorGPU:
        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, self.id * PYDTNN_OPS_EVENTS + PYDTNN_OPS_EVENT_enum.FORWARD_RESHAPE_Y)
        self.y.ary = x.ary.reshape((self.model.batch_size, *self.shape))
        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, PYDTNN_EVENT_FINISHED)
        return self.y

    def backward(self, dy: TensorGPU) -> TensorGPU:
        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, self.id * PYDTNN_OPS_EVENTS + PYDTNN_OPS_EVENT_enum.BACKWARD_RESHAPE_DX)
        self.dx = dy.reshape((self.model.batch_size, *self.prev_shape))
        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, PYDTNN_EVENT_FINISHED)
        return self.dx
