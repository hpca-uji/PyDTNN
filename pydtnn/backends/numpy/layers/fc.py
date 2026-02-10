from pydtnn.libs import numpy as np
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    import numpy as np

from pydtnn.backends.numpy.layers.layer import LayerNumpy
from pydtnn.layers.fc import FC
from pydtnn.model import Model
from pydtnn.utils.performance_models import matmul_time
from pydtnn.tracers.events import PYDTNN_OPS_EVENT, PYDTNN_OPS_EVENTS, PYDTNN_EVENT_FINISHED, PYDTNN_OPS_EVENT_enum


class FCNumpy(FC[np.ndarray], LayerNumpy):

    biases: np.ndarray

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # The following attributes will be initalized in "initalize"
        self.x: np.ndarray = None  # type: ignore
        self.dw: np.ndarray = None  # type: ignore
        self.db: np.ndarray = None  # type: ignore
    # --

    def _model_init(self, prev_shape, x=None):
        super()._model_init(prev_shape, x)
        self.weights = np.asarray(self.weights_initializer(self.weights_shape, self.model.dtype), order="C")
        self.nparams += self.weights.size
        self.memory_used += self.weights.nbytes

        # Initialize outputs:
        # NOTE: These attributes only store data, their values before the operation doesn't matter; they're initalized due avoid warnings in "LayerAndActivationBase.export".
        self.y = np.zeros((self.model.batch_size, *self.shape), dtype=self.model.dtype)
        self.memory_used += self.y.nbytes

        self.dx = np.zeros(shape=(self.model.batch_size, *self.prev_shape), dtype=self.model.dtype, order="C")
        self.dw = np.zeros(shape=self.weights_shape, dtype=self.model.dtype, order="C")
        self.memory_used += self.dx.nbytes + self.dw.nbytes

        if self.use_bias:
            self.biases = np.asarray(self.biases_initializer(self.shape, self.model.dtype), order="C")
            self.nparams += self.biases.size
            self.memory_used += self.biases.nbytes

            if not self.model.evaluate_only:
                self.db = np.zeros(self.shape, dtype=self.model.dtype)
                self.memory_used += self.db.nbytes

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
        y = self.y[: x.shape[0], :]

        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, self.id * PYDTNN_OPS_EVENTS + PYDTNN_OPS_EVENT_enum.FORWARD_MATMUL)
        np.matmul(x, self.weights, out=y,
                  dtype=self.model.dtype)

        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, PYDTNN_EVENT_FINISHED)

        if self.use_bias:
            np.add(y, self.biases, out=y,
                   dtype=self.model.dtype)

        return np.asarray(y, dtype=self.model.dtype, order="C")
    # ---

    def backward(self, dy: np.ndarray) -> np.ndarray:

        # self.model.mode = ModelModeEnum.TRAIN is asumed from this point.
        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, self.id * PYDTNN_OPS_EVENTS + PYDTNN_OPS_EVENT_enum.COMP_DW_MATMUL)
        # self.dw = np.matmul(self.x.T, dy)
        np.matmul(self.x.T, dy, self.dw,
                  dtype=self.model.dtype)
        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, PYDTNN_EVENT_FINISHED)

        if self.use_bias:
            # self.db = np.sum(dy, axis=0)
            np.sum(dy, axis=0, out=self.db)

        dx = self.dx[:self.x.shape[0], :]
        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, self.id * PYDTNN_OPS_EVENTS + PYDTNN_OPS_EVENT_enum.COMP_DX_MATMUL)
        # dx = np.matmul(dy, self.weights.T)
        np.matmul(dy, self.weights.T, out=dx,
                  dtype=self.model.dtype)
        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, PYDTNN_EVENT_FINISHED)
        return np.asarray(dx, dtype=self.model.dtype, order="C")
    # --
