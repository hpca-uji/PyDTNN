"""Numpy backend implementation for the ConcatenationBlock layer."""

import logging
from typing import TYPE_CHECKING, Any

from pydtnn.backends.numpy.layers.abstract.block_layer import AbstractBlockLayerNumpy
from pydtnn.layers.concatenation_block import ConcatenationBlock
from pydtnn.libs import numpy as np
from pydtnn.tracers.events import (PYDTNN_EVENT_FINISHED, PYDTNN_MDL_EVENT, PYDTNN_MDL_EVENTS,
                                   PYDTNN_OPS_EVENT, PYDTNN_OPS_EVENTS, MdlEventEnum, OpsEventEnum)
from pydtnn.utils.constants import ArrayShape

__all__ = ("ConcatenationBlockNumpy",)


logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    import numpy as np  # noqa: F811 (override typing)


class ConcatenationBlockNumpy(ConcatenationBlock[np.ndarray], AbstractBlockLayerNumpy):
    """Numpy-based implementation of a concatenation block layer."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Initializes the ConcatenationBlockNumpy instance."""
        super().__init__(*args, **kwargs)
        # The next attributes will be initialized later
        self.out_co: list[int] = None  # pyright: ignore[reportAttributeAccessIssue]
        self.idx_co: np.ndarray = None  # pyright: ignore[reportAttributeAccessIssue]
        self.concat_dim: int = None  # pyright: ignore[reportAttributeAccessIssue]
        self.dx: np.ndarray = None  # pyright: ignore[reportAttributeAccessIssue]

    def _model_init(self, prev_shape: ArrayShape, x: np.ndarray) -> None:
        """Initializes model-specific buffers and memory tracking."""
        super()._model_init(prev_shape, x)
        self.y: np.ndarray = np.zeros((self.model.batch_size, *self.shape), dtype=self.model.dtype)
        self.memory_used += self.y.nbytes

        dx_shape = (len(self.paths), self.model.batch_size, *self.prev_shape)
        self.dx = np.zeros(dx_shape, dtype=self.model.dtype, order="C")
        self.memory_used += self.dx.nbytes

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Performs the forward pass by concatenating outputs from multiple paths."""
        self.model.tracer.emit_event(
            PYDTNN_OPS_EVENT, self.id * PYDTNN_OPS_EVENTS + OpsEventEnum.FORWARD_REPLICATE
        )
        _x: list[np.ndarray] = [np.zeros((0,), dtype=self.model.dtype, order="C")] * len(self.paths)
        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, PYDTNN_EVENT_FINISHED)
        y = self.y[: x.shape[0], :]

        for i, p in enumerate(self.paths):
            x_forward = x[i] if self.num_inputs > 1 else x
            for layer in p:
                self.model.tracer.emit_event(
                    PYDTNN_MDL_EVENT, layer.id * PYDTNN_MDL_EVENTS + MdlEventEnum.FORWARD
                )
                x_forward = layer.forward(x_forward)
                self.model.tracer.emit_event(PYDTNN_MDL_EVENT, PYDTNN_EVENT_FINISHED)
            _x[i] = x_forward
        self.model.tracer.emit_event(
            PYDTNN_OPS_EVENT, self.id * PYDTNN_OPS_EVENTS + OpsEventEnum.FORWARD_CONCAT
        )
        np.concatenate(_x, axis=self.concat_dim, out=y)
        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, PYDTNN_EVENT_FINISHED)

        return np.asarray(y, dtype=self.model.dtype, order="C")

    def backward(self, dy: np.ndarray) -> np.ndarray:
        """Performs the backward pass by splitting gradients and propagating through paths."""
        self.model.tracer.emit_event(
            PYDTNN_OPS_EVENT, self.id * PYDTNN_OPS_EVENTS + OpsEventEnum.BACKWARD_SPLIT
        )
        real_batch_size = dy.shape[0]
        self.dx.fill(0)

        dx: list[np.ndarray] = np.split(dy, self.idx_co[:-1], axis=self.concat_dim)
        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, PYDTNN_EVENT_FINISHED)
        num_paths = len(self.paths)

        for i in range(num_paths):
            p = self.paths[i]
            dx_backward = dx[i]
            for layer in reversed(p):
                self.model.tracer.emit_event(
                    PYDTNN_MDL_EVENT, layer.id * PYDTNN_MDL_EVENTS + MdlEventEnum.BACKWARD
                )
                dx_backward = layer.backward(np.asarray(dx_backward, dtype=self.model.dtype, order="C"))
                self.model.tracer.emit_event(PYDTNN_MDL_EVENT, PYDTNN_EVENT_FINISHED)

            self.model.tracer.emit_event(
                PYDTNN_OPS_EVENT,
                self.id * PYDTNN_OPS_EVENTS + OpsEventEnum.BACKWARD_ELTW_SUM,
            )
            self.model.tracer.emit_event(PYDTNN_OPS_EVENT, PYDTNN_EVENT_FINISHED)
            self.dx[i, :real_batch_size, :] = dx_backward

        if self.num_inputs == 1:
            for i in range(num_paths):
                np.add(self.dx[0], self.dx[i], out=self.dx[0], dtype=self.model.dtype)
            output = self.dx[0, :real_batch_size, :]
        else:
            output = self.dx[:, :real_batch_size, :]

        return np.asarray(output, dtype=self.model.dtype, order="C")
