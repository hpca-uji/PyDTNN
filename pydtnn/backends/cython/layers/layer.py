from pydtnn.backends.numpy.layers.layer import LayerNumpy
from pydtnn.backends.cython.abstract.layerable import LayerableCython
import logging
logger = logging.getLogger(__name__)


class LayerCython(LayerNumpy, LayerableCython):
    ...
