from pydtnn.backends.cpu.layers.abstract.block_layer import AbstractBlockLayerCPU
from pydtnn.layers.addition_block import AdditionBlock
from pydtnn.tracers.events import PYDTNN_MDL_EVENT, PYDTNN_MDL_EVENTS, PYDTNN_OPS_EVENT, PYDTNN_OPS_EVENTS, \
    PYDTNN_EVENT_FINISHED, PYDTNN_MDL_EVENT_enum, PYDTNN_OPS_EVENT_enum
import numpy as np


class AdditionBlockCPU(AdditionBlock[np.ndarray], AbstractBlockLayerCPU):

    def forward(self, x: np.ndarray) -> np.ndarray:

        num_paths = len(self.paths)
        p = self.paths[0]
        x_forward = x
        for layer in p:
            self.model.tracer.emit_event(PYDTNN_MDL_EVENT, layer.id * PYDTNN_MDL_EVENTS + PYDTNN_MDL_EVENT_enum.FORWARD)
            x_forward = layer.forward(x_forward)
            self.model.tracer.emit_event(PYDTNN_MDL_EVENT, PYDTNN_EVENT_FINISHED)
        sum_forwards = x_forward

        for i in range(1, num_paths):
            p = self.paths[i]
            x_forward = x
            for layer in p:
                self.model.tracer.emit_event(PYDTNN_MDL_EVENT, layer.id * PYDTNN_MDL_EVENTS + PYDTNN_MDL_EVENT_enum.FORWARD)
                x_forward = layer.forward(x_forward)
                self.model.tracer.emit_event(PYDTNN_MDL_EVENT, PYDTNN_EVENT_FINISHED)
            self.model.tracer.emit_event(PYDTNN_OPS_EVENT, self.id * PYDTNN_OPS_EVENTS + PYDTNN_OPS_EVENT_enum.FORWARD_ELTW_SUM)
            np.add(sum_forwards, x_forward, out=sum_forwards,
                   dtype=self.model.dtype, order="C")
            self.model.tracer.emit_event(PYDTNN_OPS_EVENT, PYDTNN_EVENT_FINISHED)

        return np.asarray(sum_forwards, dtype=self.model.dtype, order='C', copy=None)

    def backward(self, dy: np.ndarray) -> np.ndarray:
        num_paths = len(self.paths)
        p = self.paths[0]
        dx_backward = dy
        for layer in reversed(p):
            self.model.tracer.emit_event(PYDTNN_MDL_EVENT, layer.id * PYDTNN_MDL_EVENTS + PYDTNN_MDL_EVENT_enum.FORWARD)
            dx_backward = layer.backward(dx_backward)
            self.model.tracer.emit_event(PYDTNN_MDL_EVENT, PYDTNN_EVENT_FINISHED)
        dx = dx_backward

        for i in range(1, num_paths):
            p = self.paths[i]
            dx_backward = dy
            for layer in reversed(p):
                self.model.tracer.emit_event(PYDTNN_MDL_EVENT, layer.id * PYDTNN_MDL_EVENTS + PYDTNN_MDL_EVENT_enum.BACKWARD)
                dx_backward = layer.backward(dx_backward)
                self.model.tracer.emit_event(PYDTNN_MDL_EVENT, PYDTNN_EVENT_FINISHED)

            self.model.tracer.emit_event(PYDTNN_OPS_EVENT, self.id * PYDTNN_OPS_EVENTS + PYDTNN_OPS_EVENT_enum.BACKWARD_ELTW_SUM)
            np.add(dx, dx_backward, out=dx,
                   dtype=self.model.dtype, order="C")
            self.model.tracer.emit_event(PYDTNN_OPS_EVENT, PYDTNN_EVENT_FINISHED)
        return np.asarray(dx, dtype=self.model.dtype, order='C', copy=None)
