"""Numpy backend implementation of the Multiplication layer."""

import logging
from typing import TYPE_CHECKING, Any

from pydtnn.backends.numpy.layers.abstract.layer import LayerNumpy
from pydtnn.layers.multiplication import Multiplication
from pydtnn.libs import numpy as np
from pydtnn.model.base import ModelMode
from pydtnn.utils.constants import ArrayShape

__all__ = ("MultiplicationNumpy",)

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    import numpy as np  # noqa: F811 (override typing)


class MultiplicationNumpy(Multiplication[np.ndarray], LayerNumpy):
    """Numpy-based multiplication layer for matrix operations."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Initialize the MultiplicationNumpy layer."""
        super().__init__(*args, **kwargs)
        # Following attributes will be initialized later
        self.x1: np.ndarray = None  # pyright: ignore[reportAttributeAccessIssue]
        self.x2: np.ndarray = None  # pyright: ignore[reportAttributeAccessIssue]

    def _model_init(self, prev_shape: ArrayShape, x: np.ndarray) -> None:
        """Initialize model-specific parameters and performance tracking."""
        super()._model_init(prev_shape, x)
        # Performance model
        self.fwd_time = None  # pyright: ignore[reportAttributeAccessIssue]
        self.bwd_time = None  # pyright: ignore[reportAttributeAccessIssue]

    def transpose(self, x: np.ndarray) -> np.ndarray:
        """Transpose the last two dimensions of the input array."""
        return x.swapaxes(-2, -1)

    def forward(self, x1: np.ndarray, x2: np.ndarray) -> np.ndarray:
        """Perform matrix multiplication of two inputs."""
        if self.model.mode == ModelMode.TRAIN:
            self.x1 = x1
            self.x2 = x2
        return np.matmul(x1, x2)

    def backward(self, dy: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Compute gradients with respect to the inputs."""
        dx1 = np.matmul(dy, self.transpose(self.x2))
        dx2 = np.matmul(self.transpose(self.x1), dy)
        return dx1, dx2
