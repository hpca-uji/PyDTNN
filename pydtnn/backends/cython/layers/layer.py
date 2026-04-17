import logging
logger = logging.getLogger(__name__)

from pydtnn.backends.cython.abstract.layerable import LayerableCython
from pydtnn.backends.numpy.layers.layer import LayerNumpy


class LayerCython(LayerNumpy, LayerableCython):
    ...
