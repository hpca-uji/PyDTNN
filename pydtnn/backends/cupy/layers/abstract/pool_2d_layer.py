"""
CuPy implementation of 2D pooling layers for the PyDTNN framework.
"""

import numpy as np

from pydtnn.backends.cupy.layers.abstract.layer import LayerCupy
from pydtnn.backends.numpy.layers.abstract.pool_2d_layer import AbstractPool2DLayerNumpy
from pydtnn.layers.abstract.pool_2d_layer import AbstractPool2DLayer

__all__ = ("AbstractPool2DLayerCupy",)


class AbstractPool2DLayerCupy(AbstractPool2DLayerNumpy, AbstractPool2DLayer[np.ndarray], LayerCupy):
    """
    Abstract base class for 2D pooling layers using the CuPy backend.
    """

    ...
