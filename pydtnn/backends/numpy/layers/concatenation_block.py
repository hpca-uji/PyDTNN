"""Numpy backend implementation for the ConcatenationBlock layer."""

import logging
from typing import TYPE_CHECKING, Any

from pydtnn.backends.numpy.layers.abstract.block_layer import AbstractBlockLayerNumpy
from pydtnn.layers.concatenation_block import ConcatenationBlock
from pydtnn.libs import numpy as np
from pydtnn.tracers.events import (PYDTNN_EVENT_FINISHED, PYDTNN_MDL_EVENT,
                                   PYDTNN_MDL_EVENTS, PYDTNN_OPS_EVENT, PYDTNN_OPS_EVENTS,
                                   MdlEventEnum, OpsEventEnum)
from pydtnn.utils.constants import ArrayShape

__all__ = ("ConcatenationBlockNumpy",)


logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    import numpy as np


class ConcatenationBlockNumpy(ConcatenationBlock, AbstractBlockLayerNumpy):
    """Numpy-based implementation of a concatenation block layer."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Initializes the ConcatenationBlockNumpy instance."""
        super().__init__(*args, **kwargs)
        # The next attributes will be initialized later
        self.out_co: list[int] = None  # type: ignore
        self.idx_co: np.ndarray = None  # type: ignore
        self.concat_dim: int = None  # type: ignore

    def _model_init(self, prev_shape: ArrayShape, x: np.ndarray) -> None:
        """Initializes model-specific buffers and memory tracking."""
        super()._model_init(prev_shape, x)
        self.y: np.ndarray = np.zeros((self.model.batch_size, *self.shape), dtype=self.model.dtype)
        self.memory_used += self.y.nbytes

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Performs the forward pass by concatenating outputs from multiple paths."""
        self.model.tracer.emit_event(
            PYDTNN_OPS_EVENT, self.id * PYDTNN_OPS_EVENTS + OpsEventEnum.FORWARD_REPLICATE
        )
        _x: list[np.ndarray] = [np.zeros((0,), dtype=self.model.dtype, order="C")] * len(self.paths)
        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, PYDTNN_EVENT_FINISHED)
        y = self.y[: x.shape[0], :]

        for i, p in enumerate(self.paths):
            x_forward = x
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
        dx: list[np.ndarray] = np.split(dy, self.idx_co[:-1], axis=self.concat_dim)
        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, PYDTNN_EVENT_FINISHED)

        num_paths = len(self.paths)

        p = self.paths[0]
        for layer in reversed(p):
            self.model.tracer.emit_event(
                PYDTNN_MDL_EVENT, layer.id * PYDTNN_MDL_EVENTS + MdlEventEnum.BACKWARD
            )
            dx[0] = layer.backward(np.asarray(dx[0], dtype=self.model.dtype, order="C"))
            self.model.tracer.emit_event(PYDTNN_MDL_EVENT, PYDTNN_EVENT_FINISHED)

        for i in range(1, num_paths):
            p = self.paths[i]
            for layer in reversed(p):
                self.model.tracer.emit_event(
                    PYDTNN_MDL_EVENT, layer.id * PYDTNN_MDL_EVENTS + MdlEventEnum.BACKWARD
                )
                dx[i] = layer.backward(np.asarray(dx[i], dtype=self.model.dtype, order="C"))
                self.model.tracer.emit_event(PYDTNN_MDL_EVENT, PYDTNN_EVENT_FINISHED)

            self.model.tracer.emit_event(
                PYDTNN_OPS_EVENT,
                self.id * PYDTNN_OPS_EVENTS + OpsEventEnum.BACKWARD_ELTW_SUM,
            )
            np.add(dx[0], dx[i], out=dx[0], dtype=self.model.dtype)
            self.model.tracer.emit_event(PYDTNN_OPS_EVENT, PYDTNN_EVENT_FINISHED)
        return np.asarray(dx[0], dtype=self.model.dtype, order="C")
