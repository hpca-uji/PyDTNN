import numpy as np

from pydtnn.backends.cpu.layers.abstract_block_layer_cpu import AbstractBlockLayerCPU
from pydtnn.layers import ConcatenationBlock
from pydtnn.tracers import PYDTNN_MDL_EVENT, PYDTNN_MDL_EVENTS, PYDTNN_OPS_EVENT, PYDTNN_OPS_EVENTS, \
    PYDTNN_EVENT_FINISHED, PYDTNN_MDL_EVENT_enum, PYDTNN_OPS_EVENT_enum  
from pydtnn.utils import PYDTNN_TENSOR_FORMAT

CONCAT_DIM_NCHW  = 1
CONCAT_DIM_NHWC = -1

class ConcatenationBlockCPU(AbstractBlockLayerCPU, ConcatenationBlock):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # The next attributes will be initialized later
        self.out_co:list[int] = None
        self.idx_co:np.ndarray = None
        self.concat_dim:int = None

    def initialize_block_layer(self):
        super().initialize_block_layer()        
        match self.model.tensor_format:
            case PYDTNN_TENSOR_FORMAT.NCHW:
                assert all([tuple(o[1:]) == tuple(self.out_shapes[0][1:]) for o in self.out_shapes])
                self.out_co = [s[0] for s in self.out_shapes]
                self.idx_co = np.cumsum(self.out_co, axis=0)
                self.shape = (sum(self.out_co), *self.out_shapes[0][1:])
                self.concat_dim = CONCAT_DIM_NCHW
            case PYDTNN_TENSOR_FORMAT.NHWC:
                assert all([tuple(o[:-1]) == tuple(self.out_shapes[0][:-1]) for o in self.out_shapes])
                self.out_co = [s[-1] for s in self.out_shapes]
                self.idx_co:np.ndarray = np.cumsum(self.out_co, axis=0)
                self.shape:tuple[int] = (*self.out_shapes[0][:-1], sum(self.out_co))
                self.concat_dim = CONCAT_DIM_NHWC
            case _:
                raise NotImplementedError(f"\"ConcatenationBlockCPU\" is not implemented for \"{self.model.tensor_format}\" format.")

    def forward(self, x:np.ndarray) -> np.ndarray:
        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, self.id * PYDTNN_OPS_EVENTS + PYDTNN_OPS_EVENT_enum.FORWARD_REPLICATE)
        _x = [0] * len(self.paths)
        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, PYDTNN_EVENT_FINISHED)

        for i, p in enumerate(self.paths):
            x_forward = x
            for layer in p:
                self.model.tracer.emit_event(PYDTNN_MDL_EVENT, layer.id * PYDTNN_MDL_EVENTS + PYDTNN_MDL_EVENT_enum.FORWARD)
                x_forward = layer.forward(x_forward)
                self.model.tracer.emit_event(PYDTNN_MDL_EVENT, PYDTNN_EVENT_FINISHED)
            _x[i] = x_forward
        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, self.id * PYDTNN_OPS_EVENTS + PYDTNN_OPS_EVENT_enum.FORWARD_CONCAT)
        y = np.concatenate(_x, axis=self.concat_dim)
        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, PYDTNN_EVENT_FINISHED)

        return y

    def backward(self, dy:np.ndarray) -> np.ndarray:
        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, self.id * PYDTNN_OPS_EVENTS + PYDTNN_OPS_EVENT_enum.BACKWARD_SPLIT)
        dx: list[np.ndarray] = np.split(dy, self.idx_co[:-1], axis=self.concat_dim)
        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, PYDTNN_EVENT_FINISHED)
        
        num_paths = len(self.paths)

        p = self.paths[0]
        for layer in reversed(p):
                self.model.tracer.emit_event(PYDTNN_MDL_EVENT, layer.id * PYDTNN_MDL_EVENTS + PYDTNN_MDL_EVENT_enum.BACKWARD)
                dx[0] = layer.backward(dx[0])
                self.model.tracer.emit_event(PYDTNN_MDL_EVENT, PYDTNN_EVENT_FINISHED)

        for i in range(1, num_paths):
            p = self.paths[i]
            for layer in reversed(p):
                self.model.tracer.emit_event(PYDTNN_MDL_EVENT, layer.id * PYDTNN_MDL_EVENTS + PYDTNN_MDL_EVENT_enum.BACKWARD)
                dx[i] = layer.backward(dx[i])
                self.model.tracer.emit_event(PYDTNN_MDL_EVENT, PYDTNN_EVENT_FINISHED)

            self.model.tracer.emit_event(PYDTNN_OPS_EVENT, self.id * PYDTNN_OPS_EVENTS + PYDTNN_OPS_EVENT_enum.BACKWARD_ELTW_SUM)
            dx[0] += dx[i]
            self.model.tracer.emit_event(PYDTNN_OPS_EVENT, PYDTNN_EVENT_FINISHED)
        return dx[0]
