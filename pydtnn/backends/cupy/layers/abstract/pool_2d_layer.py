from pydtnn.layers.abstract.pool_2d_layer import AbstractPool2DLayer
import numpy as np

from pydtnn.backends.cupy.layers.layer import LayerCupy
from pydtnn.backends.numpy.layers.abstract.pool_2d_layer import AbstractPool2DLayerNumpy

__all__ = (
    "AbstractPool2DLayerCupy",
)



class AbstractPool2DLayerCupy(AbstractPool2DLayerNumpy, AbstractPool2DLayer[np.ndarray], LayerCupy):
    ...
