import logging

from pydtnn.backends.direct.abstract.layerable import LayerableDirect
from pydtnn.backends.numpy.layers.layer import LayerNumpy

logger = logging.getLogger(__name__)


class LayerDirect(LayerNumpy, LayerableDirect):
    ...
