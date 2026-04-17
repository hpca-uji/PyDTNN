from pydtnn.backends.cython.layers.layer import LayerCython
from pydtnn.backends.numpy.layers.abstract.pool_2d_layer import AbstractPool2DLayerNumpy


class AbstractPool2DLayerCython(AbstractPool2DLayerNumpy, LayerCython):
    ...
