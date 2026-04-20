from pydtnn.backends.numpy.layers.layer import LayerNumpy
from pydtnn.backends.direct.abstract.layerable import LayerableDirect
import logging
logger = logging.getLogger(__name__)


class LayerDirect(LayerNumpy, LayerableDirect):
    ...
