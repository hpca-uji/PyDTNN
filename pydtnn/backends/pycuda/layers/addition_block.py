from pydtnn.tracers.events import (PYDTNN_EVENT_FINISHED, PYDTNN_MDL_EVENT,
                                   PYDTNN_MDL_EVENTS, PYDTNN_OPS_EVENT,
                                   PYDTNN_OPS_EVENTS, PYDTNN_MDL_EVENT_enum,
                                   PYDTNN_OPS_EVENT_enum)
from pydtnn.libs import cudnn as cudnn
from pydtnn.layers.addition_block import AdditionBlock
from pydtnn.backends.pycuda.utils.tensor_array import TensorArray
import logging

from pydtnn.backends.pycuda.layers.abstract.block_layer import AbstractBlockLayerPycuda

__all__ = (
    "AdditionBlockPycuda",
)

logger = logging.getLogger(__name__)


class AdditionBlockPycuda(AdditionBlock[TensorArray], AbstractBlockLayerPycuda):
    y: TensorArray

    def forward(self, x: TensorArray) -> TensorArray:
        for i, p in enumerate(self.paths):
            y_i = x
            for layer in p:
                self.model.tracer.emit_event(PYDTNN_MDL_EVENT, layer.id * PYDTNN_MDL_EVENTS + PYDTNN_MDL_EVENT_enum.FORWARD)
                y_i = layer.forward(y_i)
                self.model.tracer.emit_event(PYDTNN_MDL_EVENT, PYDTNN_EVENT_FINISHED)
            if i == 0:
                self.y = y_i
            else:
                alpha, beta = 1.0, 1.0
                self.model.tracer.emit_event(PYDTNN_OPS_EVENT,
                                             self.id * PYDTNN_OPS_EVENTS + PYDTNN_OPS_EVENT_enum.FORWARD_ELTW_SUM)
                # noinspection PyUnboundLocalVariable
                cudnn.cudnnAddTensor(self.model.cudnn_handle, alpha, y_i.desc,
                                     y_i.ptr_voidp, beta, self.y.desc, self.y.ptr_voidp)
                self.model.tracer.emit_event(PYDTNN_OPS_EVENT, PYDTNN_EVENT_FINISHED)
        return self.y

    def backward(self, dy: TensorArray) -> TensorArray:
        for i, p in enumerate(self.paths):
            dx_i = dy
            for layer in reversed(p):
                self.model.tracer.emit_event(PYDTNN_MDL_EVENT, layer.id * PYDTNN_MDL_EVENTS + PYDTNN_MDL_EVENT_enum.BACKWARD)
                dx_i = layer.backward(dx_i)
                self.model.tracer.emit_event(PYDTNN_MDL_EVENT, PYDTNN_EVENT_FINISHED)
            if i == 0:
                self.dx = dx_i
            else:
                alpha, beta = 1.0, 1.0
                self.model.tracer.emit_event(PYDTNN_OPS_EVENT,
                                             self.id * PYDTNN_OPS_EVENTS + PYDTNN_OPS_EVENT_enum.BACKWARD_ELTW_SUM)
                # noinspection PyUnboundLocalVariable
                cudnn.cudnnAddTensor(self.model.cudnn_handle, alpha, dx_i.desc,
                                     dx_i.ptr_voidp, beta, self.dx.desc, self.dx.ptr_voidp)
                self.model.tracer.emit_event(PYDTNN_OPS_EVENT, PYDTNN_EVENT_FINISHED)
        return self.dx
