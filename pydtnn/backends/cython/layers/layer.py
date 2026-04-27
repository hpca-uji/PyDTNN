import logging

from pydtnn.backends.cython.abstract.layerable import LayerableCython
from pydtnn.backends.numpy.layers.layer import LayerNumpy

logger = logging.getLogger(__name__)


class LayerCython(LayerNumpy, LayerableCython):
    ...
