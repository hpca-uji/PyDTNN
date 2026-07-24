"""NumPy backend implementation of the Fully Connected (FC) layer."""

import logging
from typing import TYPE_CHECKING, Any

from pydtnn.backends.numpy.layers.abstract.layer import LayerNumpy
from pydtnn.layers.fc import FC
from pydtnn.libs import numpy as np
from pydtnn.tracers.events import (PYDTNN_EVENT_FINISHED, PYDTNN_OPS_EVENT,
                                   PYDTNN_OPS_EVENTS, OpsEventEnum)
from pydtnn.utils.constants import ArrayShape
from pydtnn.utils.performance_models import matmul_time

__all__ = ("FCNumpy",)

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    import numpy as np  # noqa: F811 (override typing)


class FCNumpy(FC[np.ndarray], LayerNumpy):
    """Fully connected layer implementation using NumPy."""

    biases: np.ndarray

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Initialize the FCNumpy layer."""
        super().__init__(*args, **kwargs)
        # The following attributes will be initalized in "initalize"
        self.x: np.ndarray = None
        self.dw: np.ndarray = None
        self.db: np.ndarray = None

    def _model_init(self, prev_shape: ArrayShape, x: np.ndarray | None = None) -> None:
        """Initialize layer parameters, buffers, and performance models."""
        super()._model_init(prev_shape, x)
        self.weights = np.asarray(
            self.weights_initializer(self.weights_shape, self.model.param_dtype, self.model.random),
            order="C",
        )
        self.nparams += self.weights.size
        self.memory_used += self.weights.nbytes

        # Initialize outputs:
        # NOTE: These attributes only store data, their values before the
        # operation doesn't matter; they're initalized due avoid warnings in
        # "LayerAndActivationBase.export".
        self.y = np.zeros((self.model.batch_size, *self.shape), dtype=self.model.dtype)
        self.memory_used += self.y.nbytes

        self.dx = np.zeros(
            shape=(self.model.batch_size, *self.prev_shape), dtype=self.model.dtype, order="C"
        )
        self.dw = np.zeros(shape=self.weights_shape, dtype=self.model.param_dtype, order="C")
        self.memory_used += self.dx.nbytes + self.dw.nbytes

        if self.use_bias:
            self.biases = np.asarray(
                self.biases_initializer(self.shape, self.model.param_dtype, self.model.random),
                order="C",
            )
            self.nparams += self.biases.size
            self.memory_used += self.biases.nbytes

            if not self.model.evaluate_only:
                self.db = np.zeros(self.shape, dtype=self.model.param_dtype)
                self.memory_used += self.db.nbytes

        # Performance model
        self.fwd_time = self.bwd_time = np.zeros((4,), dtype=np.float32)
        self.fwd_time += matmul_time(
            m=self.model.batch_size,
            n=self.weights.shape[1],
            k=self.weights.shape[0],
            cpu_speed=self.model.cpu_speed,
            memory_bw=self.model.memory_bw,
            dtype=self.model.dtype,
        )
        self.bwd_time += matmul_time(
            m=self.weights.shape[0],
            n=self.weights.shape[1],
            k=self.model.batch_size,
            cpu_speed=self.model.cpu_speed,
            memory_bw=self.model.memory_bw,
            dtype=self.model.dtype,
        )
        self.bwd_time += matmul_time(
            m=self.model.batch_size,
            n=self.weights.shape[0],
            k=self.weights.shape[1],
            cpu_speed=self.model.cpu_speed,
            memory_bw=self.model.memory_bw,
            dtype=self.model.dtype,
        )

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Perform the forward pass of the FC layer."""
        self.x = x
        y = np.ascontiguousarray(self.y[: x.shape[0], :], dtype=self.model.dtype)
        self.model.tracer.emit_event(
            PYDTNN_OPS_EVENT, self.id * PYDTNN_OPS_EVENTS + OpsEventEnum.FORWARD_MATMUL
        )
        np.matmul(x, self.weights, out=y, dtype=self.model.dtype)

        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, PYDTNN_EVENT_FINISHED)

        if self.use_bias:
            np.add(y, self.biases, out=y, dtype=self.model.dtype)

        return np.asarray(y, dtype=self.model.dtype, order="C")

    def backward(self, dy: np.ndarray) -> np.ndarray:
        """Perform the backward pass of the FC layer."""

        # self.model.mode = ModelModeEnum.TRAIN is asumed from this point.
        self.model.tracer.emit_event(
            PYDTNN_OPS_EVENT, self.id * PYDTNN_OPS_EVENTS + OpsEventEnum.COMP_DW_MATMUL
        )
        # self.dw = np.matmul(self.x.T, dy)
        np.matmul(self.x.T, dy, self.dw, dtype=self.model.dtype)
        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, PYDTNN_EVENT_FINISHED)

        if self.use_bias:
            # self.db = np.sum(dy, axis=0)
            np.sum(dy, axis=0, out=self.db)

        dx = np.asarray(self.dx[: self.x.shape[0], :], dtype=self.model.dtype, order="C")
        self.model.tracer.emit_event(
            PYDTNN_OPS_EVENT, self.id * PYDTNN_OPS_EVENTS + OpsEventEnum.COMP_DX_MATMUL
        )
        # dx = np.matmul(dy, self.weights.T)
        np.matmul(dy, self.weights.T, out=dx, dtype=self.model.dtype)
        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, PYDTNN_EVENT_FINISHED)
        return np.asarray(dx, dtype=self.model.dtype, order="C")
