import logging

from pydtnn.backends.numpy.layers.layer import LayerNumpy
from pydtnn.backends.winograd.abstract.layerable import LayerableWinograd

__all__ = ("LayerWinograd",)

logger = logging.getLogger(__name__)


class LayerWinograd(LayerNumpy, LayerableWinograd): ...
