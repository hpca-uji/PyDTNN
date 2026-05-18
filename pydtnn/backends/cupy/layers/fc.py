"""
CuPy implementation of Fully Connected layers for the PyDTNN framework.
"""

import logging

import numpy as np
from cupy.cuda import Stream  # type: ignore

from pydtnn.backends.cupy.layers.abstract.layer import LayerCupy
from pydtnn.backends.numpy.layers.fc import FCNumpy
from pydtnn.utils.constants import ArrayShape

__all__ = ("FCCupy",)

logger = logging.getLogger(__name__)


class FCCupy(FCNumpy, LayerCupy):
    """
    Fully connected layer implementation using CuPy for GPU acceleration.
    """

    def _model_init(self, prev_shape: ArrayShape, x: np.ndarray | None = None) -> None:
        """
        Initializes the layer parameters and allocates a dedicated CUDA stream.

        Args:
            prev_shape: The shape of the input tensor from the previous layer.
            x: Optional input data for initialization.
        """
        super()._model_init(prev_shape, x)

        self.stream_2 = Stream()
