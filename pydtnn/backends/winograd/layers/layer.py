from pydtnn.backends.numpy.layers.layer import LayerNumpy
from pydtnn.backends.winograd.abstract.layerable import LayerableWinograd
import logging
logger = logging.getLogger(__name__)


class LayerWinograd(LayerNumpy, LayerableWinograd):
    ...
