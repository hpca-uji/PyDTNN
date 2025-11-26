import numpy as np

from pydtnn.backends.cpu.layers.layer import LayerCPU
from pydtnn.layers.fc import FC
from pydtnn.model import Model
from pydtnn.utils.performance_models import matmul_time
from pydtnn.tracers.events import PYDTNN_OPS_EVENT, PYDTNN_OPS_EVENTS, PYDTNN_EVENT_FINISHED, PYDTNN_OPS_EVENT_enum


class FCCPU(FC[np.ndarray], LayerCPU):

    biases: np.ndarray

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # The following attributes will be initalized in "initalize"
        self.x: np.ndarray = None  #type: ignore
        self.dw: np.ndarray = None  #type: ignore
        self.db: np.ndarray = None  #type: ignore
    # --

    def initialize(self, prev_shape, x = None):
        super().initialize(prev_shape, x)
        self.weights = self.weights_initializer(self.weights_shape, self.model.dtype)
        # Initialize outputs:
        # NOTE: These attributes only store data, their values before the operation doesn't matter; they're initalized due avoid warnings in "LayerAndActivationBase.export".
        self.db = np.zeros(self.shape, dtype=self.model.dtype, order="C")
        self.dy = np.zeros((self.model.batch_size, *self.shape), dtype=self.model.dtype, order="C")
        self.dw = np.zeros(shape=(*self.prev_shape, *self.shape), dtype=self.model.dtype, order="C")
        self.dx = np.zeros(shape=(self.model.batch_size, *self.prev_shape), dtype=self.model.dtype, order="C")

        if self.use_bias:
            self.biases = self.biases_initializer(self.shape, self.model.dtype)
        self.nparams = self.weights.size + (self.biases.size if self.use_bias else 0)
        # Performance model
        self.fwd_time = \
            matmul_time(m=self.model.batch_size, n=self.weights.shape[1], k=self.weights.shape[0],
                        cpu_speed=self.model.cpu_speed, memory_bw=self.model.memory_bw,
                        dtype=self.model.dtype)
        self.bwd_time = \
            matmul_time(m=self.weights.shape[0], n=self.weights.shape[1], k=self.model.batch_size,
                        cpu_speed=self.model.cpu_speed, memory_bw=self.model.memory_bw,
                        dtype=self.model.dtype) + \
            matmul_time(m=self.model.batch_size, n=self.weights.shape[0], k=self.weights.shape[1],
                        cpu_speed=self.model.cpu_speed, memory_bw=self.model.memory_bw,
                        dtype=self.model.dtype)  # type: ignore (It works well.)
    # ----

    def forward(self, x: np.ndarray) -> np.ndarray:
        self.x = x
        dy = self.dy[: x.shape[0], :]

        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, self.id * PYDTNN_OPS_EVENTS + PYDTNN_OPS_EVENT_enum.FORWARD_MATMUL)
        np.matmul(x, self.weights, out=dy, 
                  dtype=self.model.dtype, order="C")

        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, PYDTNN_EVENT_FINISHED)

        if self.use_bias:
            np.add(dy, self.biases, out=dy, 
                   dtype=self.model.dtype, order="C")

        return np.asarray(dy, dtype=self.model.dtype, order='C', copy=None)
    # ---

    def backward(self, dy: np.ndarray) -> np.ndarray:

        # self.model.mode = ModelModeEnum.TRAIN is asumed from this point.
        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, self.id * PYDTNN_OPS_EVENTS + PYDTNN_OPS_EVENT_enum.COMP_DW_MATMUL)
        # self.dw = np.matmul(self.x.T, dy)
        np.matmul(self.x.T, dy, self.dw, 
                  dtype=self.model.dtype, order="C")
        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, PYDTNN_EVENT_FINISHED)

        if self.use_bias:
            # self.db = np.sum(dy, axis=0)
            np.sum(dy, axis=0, out=self.db)

        dx = self.dx[:self.x.shape[0], :]
        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, self.id * PYDTNN_OPS_EVENTS + PYDTNN_OPS_EVENT_enum.COMP_DX_MATMUL)
        # dx = np.matmul(dy, self.weights.T)
        np.matmul(dy, self.weights.T, out=dx, 
                  dtype=self.model.dtype, order="C")
        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, PYDTNN_EVENT_FINISHED)
        return np.asarray(dx, dtype=self.model.dtype, order='C', copy=None)
    # --
