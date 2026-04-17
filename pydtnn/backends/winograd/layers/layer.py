import logging
logger = logging.getLogger(__name__)

from pydtnn.backends.winograd.abstract.layerable import LayerableWinograd
from pydtnn.backends.numpy.layers.layer import LayerNumpy


class LayerWinograd(LayerNumpy, LayerableWinograd):
    ...
