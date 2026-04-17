import logging
logger = logging.getLogger(__name__)

from pydtnn.backends.direct.abstract.layerable import LayerableDirect
from pydtnn.backends.numpy.layers.layer import LayerNumpy


class LayerDirect(LayerNumpy, LayerableDirect):
    ...
