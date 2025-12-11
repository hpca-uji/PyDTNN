from pydtnn.backends.gpu.layers.abstract.block_layer import AbstractBlockLayerGPU
from pydtnn.layers.addition_block import AdditionBlock
from pydtnn.tracers.events import PYDTNN_MDL_EVENT, PYDTNN_MDL_EVENTS, PYDTNN_OPS_EVENT, PYDTNN_OPS_EVENTS, \
    PYDTNN_EVENT_FINISHED, PYDTNN_MDL_EVENT_enum, PYDTNN_OPS_EVENT_enum
from pydtnn.libs import libcudnn as cudnn
from pydtnn.backends.gpu.utils.tensor_gpu import TensorGPU


class AdditionBlockGPU(AdditionBlock[TensorGPU], AbstractBlockLayerGPU):
    y: TensorGPU
    def forward(self, x: TensorGPU) -> TensorGPU:
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
                                     y_i.ptr, beta, self.y.desc, self.y.ptr)
                self.model.tracer.emit_event(PYDTNN_OPS_EVENT, PYDTNN_EVENT_FINISHED)
        return self.y

    def backward(self, dy: TensorGPU) -> TensorGPU:
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
                                     dx_i.ptr, beta, self.dx.desc, self.dx.ptr)
                self.model.tracer.emit_event(PYDTNN_OPS_EVENT, PYDTNN_EVENT_FINISHED)
        return self.dx
