"""Numpy backend implementation of the Multiplication layer."""

import logging
from typing import TYPE_CHECKING, Any

from pydtnn.backends.numpy.layers.abstract.block_layer import AbstractBlockLayerNumpy
from pydtnn.layers.multiplication_block import MultiplicationBlock
from pydtnn.libs import numpy as np
from pydtnn.tracers.events import PYDTNN_EVENT_FINISHED, PYDTNN_MDL_EVENT, PYDTNN_MDL_EVENTS, MdlEventEnum
from pydtnn.utils.constants import ArrayShape

__all__ = ("MultiplicationBlockNumpy",)

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    import numpy as np  # noqa: F811 (override typing)


class MultiplicationBlockNumpy(MultiplicationBlock[np.ndarray], AbstractBlockLayerNumpy):
    """Numpy-based multiplication layer for matrix operations."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Initialize the MultiplicationNumpy layer."""
        super().__init__(*args, **kwargs)
        # Following attributes will be initialized later
        self.x: np.ndarray = None  # pyright: ignore[reportAttributeAccessIssue]
        self.dx: np.ndarray = None  # pyright: ignore[reportAttributeAccessIssue]

    def _model_init(self, prev_shape: ArrayShape, x: np.ndarray) -> None:
        """Initialize model-specific parameters and performance tracking."""
        super()._model_init(prev_shape, x)
        self.x = np.zeros(self.prev_shape, dtype=self.model.dtype, order="C")
        self.dx = np.zeros(self.prev_shape, dtype=self.model.dtype, order="C")

    def transpose(self, x: np.ndarray) -> np.ndarray:
        """Transpose the last two dimensions of the input array."""
        return x.swapaxes(-2, -1)

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Perform matrix multiplication of two inputs."""

        for i, path in enumerate(self.paths):
            # NOTE: It's assumed that the input has an extra dimension:
            #  E.G.: Normal layer has 4 dimensions: NCHW => this layer has 5: [Paths]NCHW
            x_forward = x[i] if self.num_inputs > 1 else x
            for layer in path:
                self.model.tracer.emit_event(PYDTNN_MDL_EVENT, layer.id * PYDTNN_MDL_EVENTS + MdlEventEnum.FORWARD)
                x_forward = layer.forward(x_forward)
                self.model.tracer.emit_event(PYDTNN_MDL_EVENT, PYDTNN_EVENT_FINISHED)
            self.x[i, :] = x_forward

        y = self.x[0]
        for i in range(1, len(self.paths)):
            np.matmul(y, self.x[i], out=y)

        return y

    def backward(self, dy: np.ndarray) -> np.ndarray:
        """Compute gradients with respect to the inputs."""
        # TODO: Discover how to make this backwars for "n" paths.
        assert len(self.paths) == 2, f"Right now, the bacwards only allows 2 paths (2 != {len(self.paths)}). " \
                                     "Please, do a sequence of this layer in order to get the same result."

        x2 = self.x[1]  # self.x[-1]
        x1 = self.x[0]  # self.x[-2]
        dx2 = np.matmul(self.transpose(x1), dy)
        dx1 = np.matmul(dy, self.transpose(x2))

        self.dx[0] = dx1
        self.dx[1] = dx2

        for i in range(len(self.paths) - 1, -1, -1):
            path = self.paths[i]
            dx_backward = self.dx[i]
            for layer in reversed(path):
                self.model.tracer.emit_event(PYDTNN_MDL_EVENT, layer.id * PYDTNN_MDL_EVENTS + MdlEventEnum.BACKWARD)
                dx_backward = layer.backward(dx_backward)
                self.model.tracer.emit_event(PYDTNN_MDL_EVENT, PYDTNN_EVENT_FINISHED)
            self.dx[i] = dx_backward
        return self.dx

    def backward_alt(self, dy: np.ndarray) -> np.ndarray:
        """Compute gradients with respect to the inputs."""
        # TODO: Check if this is okay

        self.dx[-1] = self.x[-2] * dy
        for i in range(len(self.paths) - 2, 0, -1):
            self.dx[i] = self.x[i - 1] * self.dx[-1]
        self.dx[0] = self.dx[1] * self.x[-1]

        for i in range(len(self.paths) - 1, -1, -1):
            path = self.paths[i]
            dx_backward = self.dx[i]
            for layer in reversed(path):
                self.model.tracer.emit_event(PYDTNN_MDL_EVENT, layer.id * PYDTNN_MDL_EVENTS + MdlEventEnum.BACKWARD)
                dx_backward = layer.backward(dx_backward)
                self.model.tracer.emit_event(PYDTNN_MDL_EVENT, PYDTNN_EVENT_FINISHED)
            self.dx[i] = dx_backward

        return self.dx

    def backward_original(self, dy: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Compute gradients with respect to the inputs."""
        x2 = self.x[1]
        x1 = self.x[0]
        dx2 = np.matmul(self.transpose(x1), dy)
        dx1 = np.matmul(dy, self.transpose(x2))
        return dx1, dx2
